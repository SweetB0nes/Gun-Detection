from torch.utils.data import DataLoader

def build_dataloaders(
        train_ds, 
        test_ds,
        batch_size=8,
        shuffle=False,
        num_workers=1
    ):

    dl_train = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    dl_test = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    return dl_train, dl_test
