import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicPokerNetwork(nn.Module):
    """
    Neural network for Texas Hold'em that handles variable players and action sequences.
    Processes observation format from the HoldemTable environment.
    """

    def __init__(self, max_players=10, max_action_seq=100):
        super().__init__()

        # Constants based on your environment's observation structure
        self.max_players = max_players
        self.max_action_seq = max_action_seq

        # Card encoding: 0-51 cards, -1 for padding (using 53 tokens, 52 for padding)
        self.card_embedding = nn.Embedding(53, 16, padding_idx=52)

        # Player feature encoder (6 features per player: is_active, did_raise, money_invested, current_stack, is_current, is_bb)
        self.player_encoder = nn.Sequential(
            nn.Linear(6, 32), nn.ReLU(), nn.Linear(32, 32)
        )

        # Self-attention for variable number of players
        self.player_attention = nn.MultiheadAttention(
            embed_dim=32, num_heads=4, batch_first=True, dropout=0.1
        )

        # Action sequence LSTM (4 features per action: player_seat, action_idx, reward, done)
        self.action_lstm = nn.LSTM(
            input_size=4, hidden_size=64, num_layers=2, batch_first=True, dropout=0.1
        )

        # Global features encoder (7 features)
        self.global_encoder = nn.Sequential(
            nn.Linear(7, 32), nn.ReLU(), nn.Linear(32, 32)
        )

        # Card features encoder (2 hole + 5 community = 7 cards)
        self.card_features_encoder = nn.Sequential(
            nn.Linear(7 * 16, 32), nn.ReLU(), nn.Linear(32, 32)
        )

        # Fusion network - combine all encoded features
        self.fusion = nn.Sequential(
            nn.Linear(32 + 32 + 64 + 32, 256),  # cards + players + actions + global
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Actor head (policy) - 13 actions (FOLD through ALL_IN, excludes SMALL_BLIND/BIG_BLIND)
        self.actor = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 13))

        # Critic head (value)
        self.critic = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

        # Layer norm for stability
        self.layer_norm = nn.LayerNorm(128)

    def forward(self, observation_dict: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with variable-length inputs.

        observation_dict contains:
            - hole_cards: [batch, 2] (encoded as 0-51, -1 for no card)
            - community_cards: [batch, 5] (encoded as 0-51, -1 for no card)
            - player_features: [batch, max_players, 6]
            - player_mask: [batch, max_players] (1 for active player, 0 for inactive)
            - global_features: [batch, 7]
            - action_sequence: [batch, seq_len, 4]
            - action_mask: [batch, seq_len] (1 for valid action, 0 for padding)
        """
        batch_size = observation_dict["hole_cards"].shape[0]

        # 1. Process cards (2 hole cards + 5 community cards)
        hole_cards = observation_dict["hole_cards"].long()
        comm_cards = observation_dict["community_cards"].long()

        # Replace -1 with padding index (52)
        hole_cards = torch.where(hole_cards >= 0, hole_cards, 52)
        comm_cards = torch.where(comm_cards >= 0, comm_cards, 52)

        # Get embeddings for all cards
        hole_emb = self.card_embedding(hole_cards)  # [batch, 2, 16]
        comm_emb = self.card_embedding(comm_cards)  # [batch, 5, 16]

        # Combine all card embeddings (2+5=7 cards)
        all_cards = torch.cat([hole_emb, comm_emb], dim=1)  # [batch, 7, 16]
        all_cards_flat = all_cards.reshape(batch_size, -1)  # [batch, 7*16=112]
        card_features = self.card_features_encoder(all_cards_flat)  # [batch, 32]

        # 2. Process player features
        player_features = observation_dict["player_features"]
        player_mask = observation_dict["player_mask"]

        # Encode each player
        encoded_players = self.player_encoder(player_features.reshape(-1, 6)).reshape(
            batch_size, self.max_players, -1
        )  # [batch, max_players, 32]

        # Apply attention with mask
        # key_padding_mask: True for values to be ignored (inactive players)
        attended_players, _ = self.player_attention(
            encoded_players,
            encoded_players,
            encoded_players,
            key_padding_mask=(player_mask == 0),
        )

        # Pool players (masked mean)
        mask_expanded = player_mask.unsqueeze(-1)  # [batch, max_players, 1]
        player_pooled = torch.sum(attended_players * mask_expanded, dim=1)
        valid_players = torch.sum(mask_expanded, dim=1)
        player_features_out = player_pooled / (valid_players + 1e-8)  # [batch, 32]

        # 3. Process variable-length action sequence
        action_seq = observation_dict["action_sequence"]
        action_mask = observation_dict["action_mask"]

        # Handle sequences with packing
        seq_lengths = action_mask.sum(dim=1).cpu()

        if seq_lengths.sum() > 0:
            # Sort by sequence length for packing
            sorted_lengths, sorted_idx = torch.sort(seq_lengths, descending=True)
            action_seq_sorted = action_seq[sorted_idx]

            # Pack padded sequence
            packed_seq = nn.utils.rnn.pack_padded_sequence(
                action_seq_sorted, sorted_lengths, batch_first=True, enforce_sorted=True
            )

            # Process with LSTM
            packed_output, (hidden, _) = self.action_lstm(packed_seq)

            # Get last hidden state
            last_hidden = hidden[-1]  # Last layer hidden state

            # Unsort to original order
            _, unsort_idx = torch.sort(sorted_idx)
            action_features = last_hidden[unsort_idx]  # [batch, 64]
        else:
            # No actions yet, use zeros
            action_features = torch.zeros(batch_size, 64).to(card_features.device)

        # 4. Process global features
        global_features = self.global_encoder(
            observation_dict["global_features"]
        )  # [batch, 32]

        # 5. Combine all features
        combined = torch.cat(
            [
                card_features,  # 32
                player_features_out,  # 32
                action_features,  # 64
                global_features,  # 32
            ],
            dim=1,
        )  # Total: 160

        # 6. Fusion and output
        fused = self.fusion(combined)
        fused = self.layer_norm(fused)

        policy_logits = self.actor(fused)  # [batch, 8]
        value = self.critic(fused)  # [batch, 1]

        return policy_logits, value
