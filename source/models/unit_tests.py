import os
import torch
import tempfile
import unittest

from source.models.other.light_attention import LightAttentionFFN, LightAttention, LightAttentionFastKAN
from source.models.other.attention_lstm_hybrid import AttentionLstmHybrid
from source.models.other.lstm_reduction_hybrid import LstmReductionHybrid
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
        self.reduced_seq_len = 40
        self.out_channels = 10
        self.hidden_layers = HiddenLayers(HiddenLayers.Type.EXACT, [128])
        self.reduced_channels = 80

        # Parameters for FastKAN
        self.grid_min = -2
        self.grid_max = 2
        self.num_grids = 8

        # Parameters for LstmReductionHybrid
        self.lstm_hidden_size = 256
        self.lstm_num_layers = 2
        self.dropout_rate = 0.2

        # Parameters for AttentionLstmHybrid
        self.attention_num_heads = 4

        # Parameters for LightAttention
        self.kernel_size = 9
        self.conv_dropout_rate = 0.25
        self.ffn_dropout_rate = 0.3


    def test_create_max_pool_fast_kan(self):
        """Tests the creation of a MaxPoolFastKAN model."""
        model = MaxPoolFastKAN(
            in_channels=self.in_channels,
            in_seq_len=self.in_seq_len,
            out_channels=self.out_channels,
            reduced_seq_len=self.reduced_seq_len,
            hidden_layers=self.hidden_layers
        )
        self.assertIsNotNone(model)


    def test_create_avg_pool_fast_kan(self):
        """Tests the creation of an AvgPoolFastKAN model."""
        model = AvgPoolFastKAN(
            in_channels=self.in_channels,
            in_seq_len=self.in_seq_len,
            out_channels=self.out_channels,
            reduced_seq_len=self.reduced_seq_len,
            hidden_layers=self.hidden_layers
        )
        self.assertIsNotNone(model)


    def test_create_linear_fast_kan(self):
        """Tests the creation of a LinearFastKAN model."""
        model = LinearFastKAN(
            in_channels=self.in_channels,
            in_seq_len=self.in_seq_len,
            out_channels=self.out_channels,
            reduced_seq_len=self.reduced_seq_len,
            hidden_layers=self.hidden_layers
        )
        self.assertIsNotNone(model)


    def test_create_attention_fast_kan(self):
        """Tests the creation of an AttentionFastKAN model."""
        model = AttentionFastKAN(
            in_channels=self.in_channels,
            in_seq_len=self.in_seq_len,
            out_channels=self.out_channels,
            reduced_seq_len=self.reduced_seq_len,
            hidden_layers=self.hidden_layers
        )
        self.assertIsNotNone(model)


    def test_create_attention_mlp(self):
        """Tests the creation of an AttentionMLP model."""
        model = AttentionMLP(
            in_channels=self.in_channels,
            in_seq_len=self.in_seq_len,
            out_channels=self.out_channels,
            reduced_seq_len=self.reduced_seq_len,
            hidden_layers=self.hidden_layers
        )
        self.assertIsNotNone(model)


    def test_create_positional_fast_kan(self):
        """Tests the creation of a PositionalFastKAN model."""
        model = PositionalFastKAN(
            in_channels=self.in_channels,
            in_seq_len=self.in_seq_len,
            out_channels=self.out_channels,
            reduced_seq_len=self.reduced_seq_len,
            reduced_channels=self.reduced_channels,
            hidden_layers=self.hidden_layers
        )
        self.assertIsNotNone(model)


    def test_create_unet_fast_kan(self):
        """Tests the creation of a UNetFastKAN model."""
        model = UNetFastKAN(
            in_channels=self.in_channels,
            in_seq_len=self.in_seq_len,
            out_channels=self.out_channels,
            reduced_seq_len=self.reduced_seq_len,
            reduced_channels=self.reduced_channels,
            hidden_layers=self.hidden_layers
        )
        self.assertIsNotNone(model)


    def test_create_fast_kan(self):
        """Tests the creation of a FastKAN model."""
        model = FastKAN(
            in_channels=self.in_channels,
            in_seq_len=self.reduced_seq_len,  # FFN uses reduced_seq_len
            hidden_layers=self.hidden_layers,
            out_channels=self.out_channels
        )
        self.assertIsNotNone(model)


    def test_create_mlp(self):
        """Tests the creation of an MLP model."""
        model = MLP(
            in_channels=self.in_channels,
            in_seq_len=self.reduced_seq_len,  # FFN uses reduced_seq_len
            hidden_layers=self.hidden_layers,
            out_channels=self.out_channels
        )
        self.assertIsNotNone(model)


    def test_create_mlppp(self):
        """Tests the creation of an MLPpp model."""
        model = MLPpp(
            in_seq_len=self.reduced_seq_len,  # FFN uses reduced_seq_len
            hidden_layers=self.hidden_layers,
            out_channels=self.out_channels
        )
        self.assertIsNotNone(model)


    def test_create_lstm_reduction_hybrid(self):
        """Tests the creation of a LstmReductionHybrid model."""
        model = LstmReductionHybrid(
            in_channels=self.in_channels,
            in_seq_len=self.in_seq_len,
            out_channels=self.out_channels,
            reduced_seq_len=self.reduced_seq_len,
            lstm_hidden_size=self.lstm_hidden_size,
            lstm_num_layers=self.lstm_num_layers,
            dropout_rate=self.dropout_rate,
            hidden_layers=self.hidden_layers
        )
        self.assertIsNotNone(model)


    def test_create_light_attention_ffn(self):
        """Tests the creation of a LightAttentionFFN model."""
        model = LightAttentionFFN(
            in_channels=self.in_channels,
            hidden_layers=self.hidden_layers,
            ffn_dropout_rate=self.ffn_dropout_rate,
            out_channels=self.out_channels
        )
        self.assertIsNotNone(model)


    def test_create_light_attention(self):
        """Tests the creation of a LightAttention model."""
        model = LightAttention(
            in_channels=self.in_channels,
            in_seq_len=self.in_seq_len,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            conv_dropout_rate=self.conv_dropout_rate,
            hidden_layers=self.hidden_layers
        )
        self.assertIsNotNone(model)


    def test_create_light_attention_fast_kan(self):
        """Tests the creation of a LightAttentionFastKAN model."""
        model = LightAttentionFastKAN(
            in_channels=self.in_channels,
            in_seq_len=self.in_seq_len,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            conv_dropout_rate=self.conv_dropout_rate,
            hidden_layers=self.hidden_layers
        )
        self.assertIsNotNone(model)


    def test_create_attention_lstm_hybrid(self):
        """Tests the creation of an AttentionLstmHybrid model."""
        model = AttentionLstmHybrid(
            in_channels=self.in_channels,
            in_seq_len=self.in_seq_len,
            out_channels=self.out_channels,
            attention_num_heads=self.attention_num_heads,
            lstm_hidden_size=self.lstm_hidden_size,
            lstm_num_layers=self.lstm_num_layers,
            hidden_layers=self.hidden_layers
        )
        self.assertIsNotNone(model)



class TestModelSaveAndLoad(unittest.TestCase):


    def setUp(self):
        from source.models.reduced_ffn import MaxPoolFastKAN

        self.encoding_dim = 1000
        self.batch_size = 69
        self.input_seq_len = 42
        self.model = MaxPoolFastKAN(
            in_channels=self.encoding_dim,
            in_seq_len=self.input_seq_len,
            # out_channels=11,
            reduced_seq_len=22,
            hidden_layers=HiddenLayers(HiddenLayers.Type.RELATIVE, (0.06, 1))
        )


    def test_save_and_load(self):
        # Create a temporary directory for saving the model
        print()

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



class TestModelLoad(unittest.TestCase):

    def setUp(self):
        self.saved_model_path = "./data/saved_models/onehot/Debug_TestModel_FastKAN.pth"

    def test_load(self):
        loaded_model, _ = AbstractModel.load(self.saved_model_path)
        self.assertIsNotNone(loaded_model)
        print(f"Loaded model: {loaded_model}")



if __name__ == '__main__':
    unittest.main()