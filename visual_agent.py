"""Obraz RGB → CNN → GRU → Q; pamięć robocza i replay sekwencji."""
from collections import deque
from pathlib import Path
import random
import numpy as np
import torch
from torch import nn
from camera import CAMERA_ANGLE, CAMERA_DEPTH
from tensor_camera import TensorCamera, camera_pose
from episode import Action
from player import ROBOT_RADIUS, MOVE_DISTANCE, TURN_DEGREES
from torch.utils.checkpoint import checkpoint
from config.config import MODEL_PATH as DEFAULT_MODEL

CAMERA_WIDTH, CAMERA_HEIGHT = 80, 60
ACTION_COUNT = len(Action)
START_ACTION = ACTION_COUNT
MOVEMENT = (ROBOT_RADIUS, MOVE_DISTANCE, TURN_DEGREES)


def observe_camera(episode):
    return TensorCamera(episode.world, "cpu", CAMERA_WIDTH, CAMERA_HEIGHT).render([camera_pose(episode.player)])[0].numpy()


class VisualQNetwork(nn.Module):
    def __init__(self, hidden_size=1024):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 4)),
            nn.Flatten(),
            nn.Linear(64 * 3 * 4, 256),
            nn.ReLU(),
        )

        self.gru = nn.GRU(
            256 + ACTION_COUNT + 1,
            hidden_size,
            batch_first=True
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, ACTION_COUNT),
        )


    def forward(self, frames, previous_actions, hidden=None):
        # Surowe RGB [batch, time, height, width, channels].
        b, t, h, w, c = frames.shape
        x = frames.reshape(b * t, h, w, c).permute(0, 3, 1, 2)
        # Checkpoint CNN oszczędza aktywacje; nie odcina gradientu pełnego BPTT.
        chunks = []
        for chunk in x.split(64):
            encode = lambda pixels: self.encoder(pixels.float() / 255)
            chunks.append(checkpoint(encode, chunk, use_reentrant=False)
                          if torch.is_grad_enabled() else encode(chunk))
        features = torch.cat(chunks).reshape(b, t, -1)
        actions = nn.functional.one_hot(previous_actions, ACTION_COUNT + 1).float()
        sequence, hidden = self.gru(torch.cat((features, actions), dim=-1), hidden)
        return self.head(sequence), hidden


class VisualAgent:
    def __init__(self, seed=7, device='auto', hidden_size=1024, capacity=10000):
        torch.set_num_threads(1)
        torch.manual_seed(seed)
        self.device = torch.device('cuda' if device == 'auto' and torch.cuda.is_available()
                                   else 'cpu' if device == 'auto' else device)
        self.random = random.Random(seed)
        self.hidden_size = hidden_size
        self.capacity = capacity
        self.network = VisualQNetwork(hidden_size).to(self.device)
        self.target = VisualQNetwork(hidden_size).to(self.device)
        self.target.load_state_dict(self.network.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.0002)
        self.memory = deque()
        self.memory_steps = 0
        self.current = None
        self.episodes = self.env_steps = self.updates = 0
        self.curriculum = None
        self.rollouts = 0
        self.recent_group_wins = []
        self.training_selection = None
        self.training_config = {}
        self.training_config = {}
        self._camera_world = None
        self._camera = None
        self.reset_memory()

    def observe(self, episode):
        if self._camera_world is not episode.world:
            self._camera_world = episode.world
            self._camera = TensorCamera(episode.world, self.device)
        return self._camera.render([camera_pose(episode.player)])[0]

    def reset_memory(self):
        self.hidden = None
        self.previous_action = START_ACTION  # START, nie dodatkowa akcja w grze.

    def act(self, state, epsilon=0.0):
        with torch.no_grad():
            values, self.hidden = self.network(
                torch.as_tensor(state, device=self.device)[None, None],
                torch.tensor([[self.previous_action]], device=self.device), self.hidden)
        # GRU aktualizuje pamięć również przy losowej eksploracji.
        action = self.random.randrange(ACTION_COUNT) if self.random.random() < epsilon else int(values[0, 0].argmax())
        self.previous_action = action
        return action

    def act_batch(self, states, previous_actions, hidden, generators, epsilon):
        """Jeden forward CNN/GRU dla aktywnych robotów, osobny stan i RNG każdego."""
        with torch.no_grad():
            values, next_hidden = self.network(
                (states if isinstance(states, torch.Tensor) else torch.as_tensor(np.stack(states), device=self.device))[:, None],
                torch.tensor(previous_actions, device=self.device)[:, None], hidden)
            greedy = values[:, 0].argmax(-1).cpu().tolist()
        actions = [rng.randrange(ACTION_COUNT) if rng.random() < epsilon else action
                   for rng, action in zip(generators, greedy)]
        return actions, next_hidden

    def learn_episode(self, episode):
        """Dokładnie jeden pełny epizod i jeden krok optymalizatora."""
        if not episode['dones'] or not episode['dones'][-1] or any(episode['dones'][:-1]):
            raise ValueError('BPTT wymaga jednego zakończonego epizodu')
        return self._learn_episodes([episode])

    def learn_episodes(self, episodes):
        """Osobne historie GRU, średnia strat epizodów, jeden krok optymalizatora."""
        if not episodes:
            raise ValueError('BPTT wymaga co najmniej jednego przebiegu')
        for episode in episodes:
            if not episode['dones'] or not episode['dones'][-1] or any(episode['dones'][:-1]):
                raise ValueError('BPTT wymaga pełnych, zakończonych przebiegów')
        return self._learn_episodes(episodes)

    def remember(self, state, action, reward, next_state, done):
        if self.current is None:
            self.current = {'frames': [state.copy()], 'actions': [], 'rewards': [], 'dones': []}
            self.memory.append(self.current)
        self.current['frames'].append(next_state.copy())
        self.current['actions'].append(int(action))
        self.current['rewards'].append(float(reward))
        self.current['dones'].append(bool(done))
        self.memory_steps += 1
        if done:
            self.current = None
        # Usuwamy całe próby, zachowując pełny kontekst pozostałych sekwencji.
        while self.memory_steps > self.capacity and len(self.memory) > 1:
            self.memory_steps -= len(self.memory.popleft()['actions'])

    def _inputs(self, episode, start, end):
        if 'poses' in episode:
            frames = episode['camera'].render(episode['poses'][start:end])[None]
        else:
            frames = torch.as_tensor(np.stack(episode['frames'][start:end]), device=self.device)[None]
        previous = [START_ACTION if i == 0 else episode['actions'][i - 1] for i in range(start, end)]
        return frames, torch.tensor([previous], device=self.device)

    def learn(self, batch_size=4):
        # Tylko pełne życia, od zerowego stanu GRU do terminalnego przejścia.
        episodes = [ep for ep in self.memory if ep['dones'] and ep['dones'][-1]]
        if not episodes:
            return None
        return self._learn_episodes(self.random.choices(episodes, k=batch_size))

    def _learn_episodes(self, episodes):
        self.optimizer.zero_grad()
        total_loss = 0.0
        for episode in episodes:
            start, end = 0, len(episode['actions'])
            frames, previous = self._inputs(episode, start, end + 1)
            # Brak detach/burn-in/truncation na ścieżce sieci uczonej.
            values, _ = self.network(frames, previous)
            with torch.no_grad():
                target_values, _ = self.target(frames, previous)
                next_actions = values[:, 1:].detach().argmax(-1, keepdim=True)
                next_values = target_values[:, 1:].gather(-1, next_actions).squeeze(-1)
                rewards = torch.tensor([episode['rewards'][start:end]], device=self.device)
                dones = torch.tensor([episode['dones'][start:end]], device=self.device)
                expected = rewards + 0.99 * (~dones) * next_values
            actions = torch.tensor([episode['actions'][start:end]], device=self.device)
            predicted = values[:, :-1].gather(-1, actions[..., None]).squeeze(-1)
            loss = nn.functional.smooth_l1_loss(predicted, expected) / len(episodes)
            loss.backward()
            total_loss += loss.item()
        nn.utils.clip_grad_norm_(self.network.parameters(), 10)
        self.optimizer.step()
        with torch.no_grad():
            for target, source in zip(self.target.parameters(), self.network.parameters()):
                target.lerp_(source, 0.01)
        self.updates += 1
        return total_loss

    def save(self, path=DEFAULT_MODEL, metrics=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.tmp')
        torch.save({'version': 6, 'architecture': 'cnn_gru', 'hidden_size': self.hidden_size,
                    'camera': (CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_ANGLE, CAMERA_DEPTH),
                    'movement': MOVEMENT, 'actions': [a.name for a in Action],
                    'capacity': self.capacity, 'bptt': 'full_episode',
                    'weights': self.network.state_dict(), 'target': self.target.state_dict(),
                    'optimizer': self.optimizer.state_dict(), 'episodes': self.episodes,
                    'env_steps': self.env_steps, 'updates': self.updates, 'rollouts': self.rollouts,
                    'training_mode': 'stratified' if self.training_selection else 'best_of_batch',
                    'training_selection': self.training_selection, 'recent_group_wins': self.recent_group_wins,
                    'training_config': self.training_config,
                    'training_config': self.training_config,
                    'random_state': self.random.getstate(), 'torch_rng': torch.get_rng_state(),
                    'curriculum': self.curriculum, 'metrics': metrics or {}}, temporary)
        temporary.replace(path)

    @classmethod
    def load(cls, path=DEFAULT_MODEL, training=False, device='auto'):
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        if checkpoint.get('version') != 6 or checkpoint.get('architecture') != 'cnn_gru':
            raise ValueError('Model nie pasuje do ruchu ciągłego i 4 akcji. Użyj agent_continuous.pt.')
        if tuple(checkpoint['camera']) != (CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_ANGLE, CAMERA_DEPTH):
            raise ValueError('Model używa innej konfiguracji kamery')
        if tuple(checkpoint.get('movement', ())) != MOVEMENT or checkpoint.get('actions') != [a.name for a in Action]:
            raise ValueError('Model używa innych reguł ruchu lub akcji')
        agent = cls(device=device, hidden_size=checkpoint['hidden_size'],
                    capacity=checkpoint['capacity'])
        agent.network.load_state_dict(checkpoint['weights'])
        agent.target.load_state_dict(checkpoint['target'])
        agent.episodes, agent.env_steps, agent.updates = (checkpoint[k] for k in ('episodes', 'env_steps', 'updates'))
        if training:
            agent.optimizer.load_state_dict(checkpoint['optimizer'])
            agent.random.setstate(checkpoint['random_state'])
            torch.set_rng_state(checkpoint['torch_rng'])
        agent.training_selection = checkpoint.get('training_selection')
        agent.training_config = checkpoint.get('training_config', {})
        agent.training_config = checkpoint.get('training_config', {})
        agent.curriculum = checkpoint.get('curriculum')
        agent.rollouts = checkpoint.get('rollouts', agent.episodes)
        agent.recent_group_wins = checkpoint.get('recent_group_wins', [])[-20:]
        agent.network.train(training)
        return agent
