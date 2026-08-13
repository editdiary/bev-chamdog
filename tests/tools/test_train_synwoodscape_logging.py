from tools.train_synwoodscape import format_epoch_log


def test_format_epoch_log_separates_epoch_train_and_val_with_metric_directions():
    text = format_epoch_log(
        epoch=12,
        num_epochs=60,
        epoch_time=138.2,
        train_loss=0.4821,
        train_occ_loss=0.1032,
        train_vis_loss=0.7578,
        train_d_iou=0.812,
        train_o_iou=0.436,
        train_v_false_high=0.021,
        train_v_false_low=0.184,
        val_loss=0.5014,
        val_occ_loss=0.1110,
        val_vis_loss=0.7808,
        val_d_iou=0.795,
        val_o_iou=0.453,
        val_v_false_high=0.025,
        val_v_false_low=0.197,
        val_score=0.624,
        best_val_score=0.610,
        is_new_best=True,
    )

    lines = text.splitlines()
    assert len(lines) == 3
    assert lines[0] == "epoch 012/60 | time  138.2s | val_iou_mean↑ 0.624 | best_val_iou_mean↑ 0.624 | checkpoint: new best"
    assert lines[1] == (
        "  train | loss_total↓ 0.4821 | loss_occ↓ 0.1032 | loss_vis↓ 0.7578 | "
        "iou_drivable↑ 0.812 | iou_obstacle↑ 0.436 | vis_false_high↓ 0.021 | vis_false_low↓ 0.184"
    )
    assert lines[2] == (
        "  val   | loss_total↓ 0.5014 | loss_occ↓ 0.1110 | loss_vis↓ 0.7808 | "
        "iou_drivable↑ 0.795 | iou_obstacle↑ 0.453 | vis_false_high↓ 0.025 | vis_false_low↓ 0.197"
    )
