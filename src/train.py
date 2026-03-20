import logging
from src.setup import *
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s"
)


metadata_path = 'data/metadata.csv'
batch_size = 16

# Device
device = get_device()
logging.info(f"Using device: {device}")

# Data
metadata_df=load_data_csv(data_path=metadata_path)
train_split = get_data_split(df=metadata_df, selected_split='train')
test_split = get_data_split(df=metadata_df, selected_split='test')

train_dataset, test_dataset = create_dataset(train_df=train_split, test_df=test_split)
train_loader, test_loader = create_dataloaders(train_dataset=train_dataset, test_dataset=test_dataset, batch_size=batch_size)

# Train


