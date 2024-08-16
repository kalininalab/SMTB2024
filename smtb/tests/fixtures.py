import random

import pandas as pd
import pytest
import torch


@pytest.fixture
def mock_data_dir(tmp_path):
    # Create a temporary directory with mock data
    data_dir = tmp_path / "data"
    for ds in ["train", "valid", "test"]:
        ds_data_dir = data_dir / ds
        ds_data_dir.mkdir(parents=True)

        # Create a mock CSV file
        df = pd.DataFrame({"value": [0.1, 0.2, 0.3, 0.4]})
        df.to_csv(ds_data_dir / "df.csv", index=False)

        # Create mock .pt files
        for i in range(len(df)):
            torch.save(
                {"representations": {0: torch.rand((random.randint(50, 60), 320))}}, ds_data_dir / f"prot_{i}.pt"
            )
    print(data_dir)

    return data_dir
