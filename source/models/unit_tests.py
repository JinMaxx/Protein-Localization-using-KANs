import os
import torch
import tempfile
import unittest

from source.models.other.light_attention import LightAttentionFFN, LightAttentionLAMLP, LightAttentionFastKAN
from source.models.other.attention_lstm_hybrid import AttentionLstmHybridFastKAN
from source.models.other.lstm_reduction_hybrid import LstmAttentionReductionHybridFastKAN
from source.training.utils.hidden_layers import HiddenLayers
from source.models.ffn import MLPpp, MLP, FastKAN
from source.models.abstract import AbstractModel
from source.models.reduced_ffn import (
    MaxPoolFastKAN,
    AvgPoolFastKAN,
    LinearFastKAN,
    AttentionFastKAN,
    AttentionMLP,
    PositionalFastKAN,
    UNetFastKAN
)

# Remember to set the working directory to the project root containing /source/ and /data/


class TestModelCreation(unittest.TestCase):

    def setUp(self):
        """Set up the common parameters for the tests."""
        self.in_channels = 1024
        self.in_seq_len = 1023
        self.reduced_seq_len = 20

    def test_create_max_pool_fast_kan(self):
        """Tests the creation of a MaxPoolFastKAN model."""
        model = MaxPoolFastKAN(
            in_channels = self.in_channels,
            in_seq_len = self.in_seq_len
        )
        self.assertIsNotNone(model)


    def test_create_avg_pool_fast_kan(self):
        """Tests the creation of an AvgPoolFastKAN model."""
        model = AvgPoolFastKAN(
            in_channels = self.in_channels,
            in_seq_len = self.in_seq_len
        )
        self.assertIsNotNone(model)


    def test_create_linear_fast_kan(self):
        """Tests the creation of a LinearFastKAN model."""
        model = LinearFastKAN(
            in_channels = self.in_channels,
            in_seq_len = self.in_seq_len
        )
        self.assertIsNotNone(model)


    def test_create_attention_fast_kan(self):
        """Tests the creation of an AttentionFastKAN model."""
        model = AttentionFastKAN(
            in_channels = self.in_channels,
            in_seq_len = self.in_seq_len
        )
        self.assertIsNotNone(model)


    def test_create_attention_mlp(self):
        """Tests the creation of an AttentionMLP model."""
        model = AttentionMLP(
            in_channels = self.in_channels,
            in_seq_len = self.in_seq_len
        )
        self.assertIsNotNone(model)


    def test_create_positional_fast_kan(self):
        """Tests the creation of a PositionalFastKAN model."""
        model = PositionalFastKAN(
            in_channels = self.in_channels,
            in_seq_len = self.in_seq_len
        )
        self.assertIsNotNone(model)


    def test_create_unet_fast_kan(self):
        """Tests the creation of a UNetFastKAN model."""
        model = UNetFastKAN(
            in_channels = self.in_channels,
            in_seq_len = self.in_seq_len
        )
        self.assertIsNotNone(model)


    def test_create_fast_kan(self):
        """Tests the creation of a FastKAN model."""
        model = FastKAN(
            in_channels = self.in_channels,
            in_seq_len = self.reduced_seq_len  # FFN uses reduced_seq_len
        )
        self.assertIsNotNone(model)


    def test_create_mlp(self):
        """Tests the creation of an MLP model."""
        model = MLP(
            in_channels = self.in_channels,
            in_seq_len = self.reduced_seq_len  # FFN uses reduced_seq_len
        )
        self.assertIsNotNone(model)


    def test_create_mlppp(self):
        """Tests the creation of an MLPpp model."""
        model = MLPpp(in_seq_len=self.in_channels)
        self.assertIsNotNone(model)


    def test_create_lstm_reduction_hybrid(self):
        """Tests the creation of a LstmReductionHybrid model."""
        model = LstmAttentionReductionHybridFastKAN(
            in_channels = self.in_channels,
            in_seq_len = self.in_seq_len
        )
        self.assertIsNotNone(model)


    def test_create_light_attention_ffn(self):
        """Tests the creation of a LightAttentionFFN model."""
        model = LightAttentionFFN(
            in_channels = self.in_channels,
            in_seq_len = self.in_seq_len
        )
        self.assertIsNotNone(model)


    def test_create_light_attention(self):
        """Tests the creation of a LightAttention model."""
        model = LightAttentionLAMLP(
            in_channels = self.in_channels,
            in_seq_len = self.in_seq_len
        )
        self.assertIsNotNone(model)


    def test_create_light_attention_fast_kan(self):
        """Tests the creation of a LightAttentionFastKAN model."""
        model = LightAttentionFastKAN(
            in_channels = self.in_channels,
            in_seq_len = self.in_seq_len
        )
        self.assertIsNotNone(model)


    def test_create_attention_lstm_hybrid(self):
        """Tests the creation of an AttentionLstmHybrid model."""
        model = AttentionLstmHybridFastKAN(
            in_channels = self.in_channels,
            in_seq_len = self.in_seq_len
        )
        self.assertIsNotNone(model)



class TestModelSaveAndLoad(unittest.TestCase):


    def setUp(self):
        from source.models.reduced_ffn import MaxPoolFastKAN
        self.model = MaxPoolFastKAN(
            in_channels = 1024,
            in_seq_len = 1023,
            reduced_seq_len = 20,
            hidden_layers = HiddenLayers(HiddenLayers.Type.RELATIVE, (0.06, 1))
        )


    def test_save_and_load(self):
        print()
        # Create a temporary directory for saving the model
        with tempfile.TemporaryDirectory() as temp_dir:

            print(f"temp_dir: {temp_dir}")

            # Save the model
            saved_model_path = self.model.save(temp_dir, identifier="test_model")

            print(f"saved_model: {saved_model_path}")
            saved_model_path = os.path.abspath(saved_model_path)
            print(f"saved_model (abs): {saved_model_path}")

            # Verify the file exists
            saved_files = os.listdir(temp_dir)
            print(f"Saved files: {saved_files}")

            for file_path in saved_files:
                file_path = os.path.join(temp_dir, file_path)
                print(f"file_path: {file_path}")
                self.assertEqual(file_path, saved_model_path)

            loaded_model, _ = AbstractModel.load(saved_model_path)

            # Check that the states match
            for p1, p2 in zip(self.model.parameters(), loaded_model.parameters()):
                self.assertTrue(torch.equal(p1, p2), "Model parameters do not match after loading.")



if __name__ == '__main__':
    unittest.main()