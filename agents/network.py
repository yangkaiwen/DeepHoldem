import torch
import torch.nn as nn
import torch.nn.functional as F


class PokerFeatureExtractor(nn.Module):
    """
    Feature extractor for Texas Hold'em that handles variable players and action sequences.
    Processes observation format from the HoldemTable environment.
    """

    def __init__(
        self,
        max_players=10,
        max_action_seq=100,
        use_player_features=True,
        use_action_features=True,
    ):
        super().__init__()

        # Constants based on your environment's observation structure
        self.max_players = max_players
        self.max_action_seq = max_action_seq
        self.use_player_features = use_player_features
        self.use_action_features = use_action_features

        # Card encoding: 0-51 cards, -1 for padding (using 53 tokens, 52 for padding)
        self.card_embedding = nn.Embedding(53, 64, padding_idx=52)

        if self.use_player_features:
            # Player feature encoder (6 features per player: is_active, did_raise, money_invested, current_stack, is_current, is_bb)
            self.player_encoder = nn.Sequential(
                nn.Linear(6, 128), nn.ReLU(), nn.Linear(128, 128)
            )

            # Self-attention for variable number of players
            self.player_attention = nn.MultiheadAttention(
                embed_dim=128, num_heads=8, batch_first=True, dropout=0.1
            )
        else:
            self.player_encoder = None
            self.player_attention = None

        if self.use_action_features:
            # Action sequence LSTM (4 features per action: player_seat, action_idx, reward, done)
            self.action_lstm = nn.LSTM(
                input_size=4,
                hidden_size=256,
                num_layers=2,
                batch_first=True,
                dropout=0.1,
            )
        else:
            self.action_lstm = None

        # Global features encoder (7 features)
        self.global_encoder = nn.Sequential(
            nn.Linear(7, 128), nn.ReLU(), nn.Linear(128, 128)
        )

        # Card features encoder (2 hole + 5 community = 7 cards)
        self.card_features_encoder = nn.Sequential(
            nn.Linear(7 * 64, 128), nn.ReLU(), nn.Linear(128, 128)
        )

        # Fusion network - combine all encoded features
        fusion_input_size = (
            128
            + (128 if self.use_player_features else 0)
            + (256 if self.use_action_features else 0)
            + 128
        )
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_size, 512),  # cards + players + actions + global
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Layer norm for stability
        self.layer_norm = nn.LayerNorm(256)

    def forward(self, observation_dict: dict) -> torch.Tensor:
        """
        Forward pass with variable-length inputs.
        Returns the fused feature vector.
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
        if self.use_player_features:
            player_features = observation_dict["player_features"]
            player_mask = observation_dict["player_mask"]

            # Encode each player
            encoded_players = self.player_encoder(
                player_features.reshape(-1, 6)
            ).reshape(
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
        else:
            # If player features are disabled, we don't include them in the concatenation
            # or we could use a zero tensor if we wanted to keep the architecture fixed,
            # but here we are changing the fusion layer input size.
            pass

        # 3. Process variable-length action sequence
        if self.use_action_features:
            action_seq = observation_dict["action_sequence"]
            action_mask = observation_dict["action_mask"]

            # Handle sequences with packing
            seq_lengths = action_mask.sum(dim=1).cpu()

            # Initialize action_features with zeros
            action_features = torch.zeros(batch_size, 256).to(card_features.device)

            # Only process sequences with length > 0
            valid_indices = torch.nonzero(seq_lengths > 0).reshape(-1)

            if valid_indices.numel() > 0:
                # Select valid sequences
                valid_seqs = action_seq[valid_indices]
                valid_lengths = seq_lengths[valid_indices]

                # Sort by sequence length for packing
                sorted_lengths, sorted_idx = torch.sort(valid_lengths, descending=True)
                action_seq_sorted = valid_seqs[sorted_idx]

                # Pack padded sequence
                packed_seq = nn.utils.rnn.pack_padded_sequence(
                    action_seq_sorted,
                    sorted_lengths,
                    batch_first=True,
                    enforce_sorted=True,
                )

                # Process with LSTM
                packed_output, (hidden, _) = self.action_lstm(packed_seq)

                # Get last hidden state
                last_hidden = hidden[-1]  # Last layer hidden state

                # Unsort to original order
                _, unsort_idx = torch.sort(sorted_idx)
                valid_features = last_hidden[unsort_idx]  # [num_valid, 64]

                # Scatter back to full batch
                action_features[valid_indices] = valid_features
        else:
            pass

        # 4. Process global features
        global_features = self.global_encoder(
            observation_dict["global_features"]
        )  # [batch, 32]

        # 5. Combine all features
        features_list = [
            card_features,  # 128
        ]
        if self.use_player_features:
            features_list.append(player_features_out)  # 128
        if self.use_action_features:
            features_list.append(action_features)  # 256

        features_list.append(global_features)  # 128

        combined = torch.cat(features_list, dim=1)

        # 6. Fusion and output
        fused = self.fusion(combined)
        fused = self.layer_norm(fused)

        return fused


class ActorNetwork(nn.Module):
    def __init__(self, max_players=10, max_action_seq=100):
        super().__init__()
        self.feature_extractor = PokerFeatureExtractor(max_players, max_action_seq)
        # Actor head (policy) - 13 actions (FOLD through ALL_IN, excludes SMALL_BLIND/BIG_BLIND)
        self.actor = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 13))

    def forward(self, observation_dict):
        features = self.feature_extractor(observation_dict)
        return self.actor(features)


class CriticNetwork(nn.Module):
    def __init__(self, max_players=10, max_action_seq=100):
        super().__init__()
        self.feature_extractor = PokerFeatureExtractor(max_players, max_action_seq)
        # Critic head (value)
        self.critic = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, observation_dict):
        features = self.feature_extractor(observation_dict)
        return self.critic(features)


class DynamicPokerNetwork(nn.Module):
    """
    Legacy wrapper or combined network if needed.
    But for separate training, we should use ActorNetwork and CriticNetwork directly.
    This class is kept for compatibility if something else imports it,
    but it will now instantiate the separate networks internally if we wanted to keep the API.
    However, the user asked to separate them.
    """

    def __init__(self, max_players=10, max_action_seq=100):
        super().__init__()
        self.actor_net = ActorNetwork(max_players, max_action_seq)
        self.critic_net = CriticNetwork(max_players, max_action_seq)

    def forward(self, observation_dict):
        policy_logits = self.actor_net(observation_dict)
        values = self.critic_net(observation_dict)
        return policy_logits, values
