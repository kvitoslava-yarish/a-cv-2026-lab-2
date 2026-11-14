from src.data import find_voc_root


def make_voc(path, splits, with_seg):
    (path / "ImageSets" / "Segmentation").mkdir(parents=True)
    (path / "JPEGImages").mkdir()
    for s in splits:
        (path / "ImageSets" / "Segmentation" / f"{s}.txt").write_text("x\n")
    if with_seg:
        (path / "SegmentationClass").mkdir()
        (path / "SegmentationObject").mkdir()


def test_finds_kaggle_train_val_layout(tmp_path):
    ds = tmp_path / "pascal-voc-2012-dataset"
    make_voc(ds / "VOC2012_test" / "VOC2012_test", ["test", "train"], with_seg=False)
    train_val = ds / "VOC2012_train_val" / "VOC2012_train_val"
    make_voc(train_val, ["train", "val", "trainval"], with_seg=True)
    assert find_voc_root({"voc_root": None}, bases=[tmp_path]) == train_val


def test_explicit_root_wins(tmp_path):
    assert find_voc_root({"voc_root": str(tmp_path)}, bases=[]) == tmp_path
