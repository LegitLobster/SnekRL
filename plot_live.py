import argparse
import csv
import json
import socket
from collections import deque
from pathlib import Path

from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg


def read_csv(path):
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="ascii") as f:
        reader = csv.DictReader(f)
        return list(reader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, default="rl_out/train_log.csv")
    parser.add_argument("--eval", type=str, default="rl_out/eval_log.csv")
    parser.add_argument("--refresh-ms", type=int, default=16)
    parser.add_argument("--stream-max-rows", type=int, default=5)
    parser.add_argument("--use-opengl", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--series-len", type=int, default=0)
    parser.add_argument("--display-max-points", type=int, default=0)
    # Back-compat for older launchers; ignored.
    parser.add_argument("--display-window", type=int, default=0)
    parser.add_argument("--stop-file", type=str, default="")
    parser.add_argument("--stream-port", type=int, default=8765)
    args = parser.parse_args()

    log_path = Path(args.log)
    eval_path = Path(args.eval)
    stop_path = Path(args.stop_file) if args.stop_file else None

    pg.setConfigOption("background", "w")
    pg.setConfigOption("foreground", "k")
    pg.setConfigOption("antialias", False)
    pg.setConfigOption("useOpenGL", args.use_opengl)

    app = QtWidgets.QApplication([])
    win = QtWidgets.QWidget()
    version_tag = "plot_live v2026-01-26d"
    win.setWindowTitle(f"Snek RL Training ({version_tag})")
    layout = QtWidgets.QGridLayout(win)

    plots = []
    for r in range(4):
        for c in range(3):
            pw = pg.PlotWidget()
            pw.showGrid(x=True, y=True, alpha=0.3)
            pw.setClipToView(False)
            pw.setDownsampling(auto=False)
            pw.enableAutoRange(x=True, y=True)
            layout.addWidget(pw, r, c)
            plots.append(pw)
            pw.addLegend(offset=(10, 10))

    clock_label = QtWidgets.QLabel("Runtime: 00:00:00")
    layout.addWidget(clock_label, 4, 0, 1, 2)
    stop_btn = QtWidgets.QPushButton("Stop")
    layout.addWidget(stop_btn, 4, 2)
    status_label = QtWidgets.QLabel("Stream: waiting...")
    layout.addWidget(status_label, 5, 0, 1, 3)

    def on_stop():
        if stop_path is not None:
            stop_path.write_text("stop", encoding="ascii")

    stop_btn.clicked.connect(on_stop)

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
    server.bind(("127.0.0.1", args.stream_port))
    server.listen(1)
    server.setblocking(False)
    connections = []
    buffers = {}
    stream_rows = []
    series_len = args.series_len
    if series_len and series_len > 0:
        def _series():
            return deque(maxlen=series_len)
    else:
        def _series():
            return []

    series = {
        "steps": _series(),
        "mean_reward": _series(),
        "mean_len": _series(),
        "mean_max_len": _series(),
        "mean_reward_long": _series(),
        "mean_len_long": _series(),
        "mean_max_len_long": _series(),
        "mean_loss_long": _series(),
        "loss": _series(),
        "fps": _series(),
        "eps": _series(),
        "best_train_max_len": _series(),
        "best_eval_max_len": _series(),
        "board_size": _series(),
        "runtime_sec": _series(),
        "best_train_rate_per_min": _series(),
        "best_eval_rate_per_min": _series(),
    }
    cached_log_rows = []
    last_csv_read = 0
    last_stream_time = 0
    using_stream = False
    last_data_time = 0
    total_stream_rows = 0

    def accept_new():
        try:
            conn, _addr = server.accept()
            conn.setblocking(False)
            connections.append(conn)
            buffers[conn] = ""
        except BlockingIOError:
            pass

    def read_stream():
        for conn in list(connections):
            while True:
                try:
                    data = conn.recv(16384)
                except BlockingIOError:
                    break
                except OSError:
                    connections.remove(conn)
                    buffers.pop(conn, None)
                    break
                if not data:
                    connections.remove(conn)
                    buffers.pop(conn, None)
                    break
                text = buffers.get(conn, "") + data.decode("ascii", errors="ignore")
                lines = text.split("\n")
                buffers[conn] = lines[-1]
                for line in lines[:-1]:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        stream_rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    colors = {
        "regular": (31, 119, 180),
        "long_start": (255, 127, 14),
        "long": (255, 127, 14),
        "loss": (31, 119, 180),
        "loss_long": (255, 127, 14),
        "fps": (31, 119, 180),
        "eps": (31, 119, 180),
        "eval": (31, 119, 180),
        "best_train": (31, 119, 180),
        "best_eval": (255, 127, 14),
        "board": (31, 119, 180),
        "runtime": (31, 119, 180),
    }

    def pen_for(label):
        color = colors.get(label, (31, 119, 180))
        return pg.mkPen(color=color, width=2)

    plot_defs = {
        0: [("regular", "Train mean reward"), ("long_start", "Train mean reward")],
        1: [("regular", "Train episode duration (steps)"), ("long_start", "Train episode duration (steps)")],
        2: [("regular", "Train mean max snake length (exploring)"), ("long", "Train mean max snake length (exploring)")],
        3: [("loss", "Loss"), ("loss_long", "Loss")],
        4: [("fps", "FPS")],
        5: [("eps", "Epsilon")],
        6: [("eval", "Eval max snake length (greedy)")],
        7: [("eval", "Eval mean reward (greedy)")],
        8: [("best_train", "Best max snake length (train vs eval)"), ("best_eval", "Best max snake length (train vs eval)")],
        9: [("board", "Board size")],
        10: [("runtime", "Runtime (min)")],
        11: [("best_train", "Best max len ?/min"), ("best_eval", "Best max len ?/min")],
    }

    curves = {}
    for idx, lines in plot_defs.items():
        plots[idx].setTitle(lines[0][1])
        for label, _title in lines:
            curves[(idx, label)] = plots[idx].plot([], [], name=label, pen=pen_for(label))

    eval_cache = {"steps": [], "max_len": [], "reward": []}
    last_eval_read = 0

    def set_curve(curve, x, y):
        curve.setData(x, y, skipFiniteCheck=True)

    def downsample(xs, ys):
        max_pts = args.display_max_points
        n = min(len(xs), len(ys))
        if max_pts is None or max_pts <= 0 or n <= max_pts:
            return xs[:n], ys[:n]
        stride = max(1, n // max_pts)
        return xs[:n:stride], ys[:n:stride]

    def append_value(key, value):
        series[key].append(value)

    def append_row(row):
        if "steps" not in row:
            return
        try:
            step = int(row["steps"])
        except (TypeError, ValueError):
            return
        append_value("steps", step)

        def to_float(val, default=0.0):
            try:
                return float(val)
            except (TypeError, ValueError):
                return default

        raw = {
            "mean_reward": to_float(row.get("mean_reward")),
            "mean_len": to_float(row.get("mean_len")),
            "mean_max_len": to_float(row.get("mean_max_len")),
            "mean_reward_long": to_float(row.get("mean_reward_long")),
            "mean_len_long": to_float(row.get("mean_len_long")),
            "mean_max_len_long": to_float(row.get("mean_max_len_long")),
            "mean_loss_long": to_float(row.get("mean_loss_long"), 0.0),
            "loss": to_float(row.get("loss"), 0.0),
            "fps": to_float(row.get("fps"), 0.0),
            "eps": to_float(row.get("eps"), 0.0),
        }
        for k, v in raw.items():
            append_value(k, v)
        append_value("best_train_max_len", to_float(row.get("best_train_max_len"), 0.0))
        append_value("best_eval_max_len", to_float(row.get("best_eval_max_len"), 0.0))

        training_started = row.get("training_started", True)
        if isinstance(training_started, str):
            training_started = training_started.strip().lower() in ("1", "true", "yes", "y")
        training_started = bool(training_started)
        if training_started:
            bt = to_float(row.get("best_train_rate_per_min"), 0.0)
            be = to_float(row.get("best_eval_rate_per_min"), 0.0)
            append_value("best_train_rate_per_min", bt)
            append_value("best_eval_rate_per_min", be)

        board_val = row.get("board_size", "")
        if isinstance(board_val, str) and "x" in board_val:
            try:
                board_val = int(board_val.split("x")[0])
            except ValueError:
                board_val = 0.0
        append_value("board_size", to_float(board_val, 0.0))
        append_value("runtime_sec", to_float(row.get("runtime_sec"), 0.0))

    frame_idx = 0

    def update():
        nonlocal last_eval_read, last_csv_read, cached_log_rows, last_stream_time, using_stream, total_stream_rows, last_data_time, frame_idx
        accept_new()
        read_stream()

        if stream_rows:
            total_stream_rows += len(stream_rows)
            rows = stream_rows
            if len(rows) > args.stream_max_rows:
                rows = rows[-args.stream_max_rows :]
            for row in rows:
                append_row(row)
            stream_rows.clear()
            last_stream_time = QtCore.QTime.currentTime().msecsSinceStartOfDay()
            using_stream = True
            last_data_time = last_stream_time
        else:
            now_ms = QtCore.QTime.currentTime().msecsSinceStartOfDay()
            stream_stale = (not using_stream) or (now_ms - last_stream_time > 2000)
            needs_seed = not series["steps"]
            if (stream_stale or needs_seed) and now_ms - last_csv_read > 500:
                cached_log_rows = read_csv(log_path)
                last_csv_read = now_ms
                if cached_log_rows:
                    for key in series:
                        series[key].clear()
                    for row in cached_log_rows:
                        append_row(row)
                    last_data_time = now_ms

        now_ms = QtCore.QTime.currentTime().msecsSinceStartOfDay()
        if now_ms - last_eval_read > 1000:
            eval_rows = read_csv(eval_path)
            eval_cache["steps"] = [int(r["steps"]) for r in eval_rows if r.get("steps")]
            eval_cache["max_len"] = [int(r["max_len"]) for r in eval_rows if r.get("max_len")]
            eval_cache["reward"] = [float(r["mean_eval_reward"]) for r in eval_rows if r.get("mean_eval_reward")]
            last_eval_read = now_ms

        steps_all = list(series["steps"])
        if steps_all:
            steps = steps_all
            mean_reward = list(series["mean_reward"])
            mean_reward_long = list(series["mean_reward_long"])
            mean_len = list(series["mean_len"])
            mean_len_long = list(series["mean_len_long"])
            mean_max_len = list(series["mean_max_len"])
            mean_max_len_long = list(series["mean_max_len_long"])
            loss = list(series["loss"])
            mean_loss_long = list(series["mean_loss_long"])
            fps = list(series["fps"])
            eps = list(series["eps"])
            best_train_max_len = list(series["best_train_max_len"])
            best_eval_max_len = list(series["best_eval_max_len"])
            board_size = list(series["board_size"])
            runtime_sec = list(series["runtime_sec"])
            best_train_rate = list(series["best_train_rate_per_min"])
            best_eval_rate = list(series["best_eval_rate_per_min"])

            x0, y0 = downsample(steps[: len(mean_reward)], mean_reward)
            set_curve(curves[(0, "regular")], x0, y0)
            if mean_reward_long:
                x1, y1 = downsample(steps[: len(mean_reward_long)], mean_reward_long)
                set_curve(curves[(0, "long_start")], x1, y1)
            x2, y2 = downsample(steps[: len(mean_len)], mean_len)
            set_curve(curves[(1, "regular")], x2, y2)
            if mean_len_long:
                x3, y3 = downsample(steps[: len(mean_len_long)], mean_len_long)
                set_curve(curves[(1, "long_start")], x3, y3)
            x4, y4 = downsample(steps[: len(mean_max_len)], mean_max_len)
            set_curve(curves[(2, "regular")], x4, y4)
            if mean_max_len_long:
                x5, y5 = downsample(steps[: len(mean_max_len_long)], mean_max_len_long)
                set_curve(curves[(2, "long")], x5, y5)

            x6, y6 = downsample(steps[: len(loss)], loss)
            set_curve(curves[(3, "loss")], x6, y6)
            if mean_loss_long:
                x7, y7 = downsample(steps[: len(mean_loss_long)], mean_loss_long)
                set_curve(curves[(3, "loss_long")], x7, y7)
            x8, y8 = downsample(steps[: len(fps)], fps)
            set_curve(curves[(4, "fps")], x8, y8)
            x9, y9 = downsample(steps[: len(eps)], eps)
            set_curve(curves[(5, "eps")], x9, y9)

            if eval_cache["steps"] and eval_cache["max_len"]:
                eval_x = eval_cache["steps"][: len(eval_cache["max_len"])]
                x10, y10 = downsample(eval_x, eval_cache["max_len"])
                set_curve(curves[(6, "eval")], x10, y10)
            if eval_cache["steps"] and eval_cache["reward"]:
                eval_x_r = eval_cache["steps"][: len(eval_cache["reward"])]
                x11, y11 = downsample(eval_x_r, eval_cache["reward"])
                set_curve(curves[(7, "eval")], x11, y11)

            x12, y12 = downsample(steps[: len(best_train_max_len)], best_train_max_len)
            set_curve(curves[(8, "best_train")], x12, y12)
            x13, y13 = downsample(steps[: len(best_eval_max_len)], best_eval_max_len)
            set_curve(curves[(8, "best_eval")], x13, y13)
            x14, y14 = downsample(steps[: len(board_size)], board_size)
            set_curve(curves[(9, "board")], x14, y14)

            runtime_min = [v / 60.0 for v in runtime_sec]
            x15, y15 = downsample(steps[: len(runtime_min)], runtime_min)
            set_curve(curves[(10, "runtime")], x15, y15)
            rate_steps = steps[-len(best_train_rate) :] if best_train_rate else []
            x16, y16 = downsample(rate_steps, best_train_rate)
            set_curve(curves[(11, "best_train")], x16, y16)
            rate_steps_eval = steps[-len(best_eval_rate) :] if best_eval_rate else []
            x17, y17 = downsample(rate_steps_eval, best_eval_rate)
            set_curve(curves[(11, "best_eval")], x17, y17)

            if runtime_sec:
                total = int(runtime_sec[-1])
                hours = total // 3600
                mins = (total % 3600) // 60
                secs = total % 60
                clock_label.setText(f"Runtime: {hours:02d}:{mins:02d}:{secs:02d}")
        else:
            if last_data_time:
                secs = int(last_data_time / 1000)
                hours = secs // 3600
                mins = (secs % 3600) // 60
                secs = secs % 60
                clock_label.setText(f"Runtime: {hours:02d}:{mins:02d}:{secs:02d}")

        now_ms = QtCore.QTime.currentTime().msecsSinceStartOfDay()
        age = now_ms - last_stream_time if last_stream_time else -1
        status_label.setText(
            f"Stream rows: {total_stream_rows} | last stream age ms: {age} | conns: {len(connections)}"
            f" | points: {len(series['steps'])} | maxPts={args.display_max_points}"
            f" | opengl={args.use_opengl}"
        )

    timer = QtCore.QTimer()
    timer.setTimerType(QtCore.Qt.PreciseTimer)
    timer.timeout.connect(update)
    timer.start(args.refresh_ms)

    win.show()
    update()
    app.exec_()


if __name__ == "__main__":
    main()
