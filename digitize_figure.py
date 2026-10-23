"""Mouse-driven digitizer for the project's Fig. 3c image."""

import argparse
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGE = SCRIPT_DIR / "fig3c.png"
DEFAULT_OUTPUT = SCRIPT_DIR / "data" / "digitized3cdata" / "fig3c_digitized.csv"

SERIES = (
    "calorimetry-derived DeltaS",
    "directly measured",
    "C-C-derived DeltaS",
)

REFERENCE_IMAGE_SIZE = (1270, 1032)
X_TICK_PIXELS = (187.0, 1181.0)
Y_TICK_PIXELS = (871.0, 29.0)


def collect_clicks(ax, fig, count, title):
    ax.set_title(title)
    fig.canvas.draw_idle()
    plt.pause(0.1)
    clicks = np.asarray(plt.ginput(count, timeout=0), dtype=float)
    expected_shape = (count, 2) if count > 0 else None
    if clicks.size == 0 and count < 0:
        return np.empty((0, 2), dtype=float)
    if expected_shape is not None and clicks.shape != expected_shape:
        raise RuntimeError(
            f"Expected {count} clicks, received {len(clicks)}. "
            "Close the figure to cancel and rerun the script."
        )
    return clicks.reshape(-1, 2)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Digitize the three plotted series in Fig. 3c."
    )
    parser.add_argument("image", nargs="?", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    if argv is None and "ipykernel" in sys.modules:
        argv = []
    args = parser.parse_args(argv)

    image_path = args.image if args.image.is_absolute() else Path.cwd() / args.image
    output_path = (
        args.output if args.output.is_absolute() else SCRIPT_DIR / args.output
    )
    if not image_path.is_file():
        parser.error(f"Image file not found: {image_path}")

    image = mpimg.imread(image_path)
    image_height, image_width = image.shape[:2]
    scale_x = image_width / REFERENCE_IMAGE_SIZE[0]
    scale_y = image_height / REFERENCE_IMAGE_SIZE[1]
    x_tick_pixels = np.asarray(X_TICK_PIXELS) * scale_x
    y_tick_pixels = np.asarray(Y_TICK_PIXELS) * scale_y

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.imshow(image)
    ax.axis("off")
    fig.suptitle("Fig. 3c digitizer - click the requested locations in the image")
    plt.show(block=False)
    plt.pause(0.5)

    rows = []
    for series in SERIES:
        points = collect_clicks(
            ax,
            fig,
            -1,
            f"Click every marker for {series}; press Enter when done",
        )
        if not len(points):
            raise RuntimeError(
                f"No points selected for '{series}'. Close the figure to cancel."
            )

        temperatures = (points[:, 0] - x_tick_pixels[0]) * 400.0 / (
            x_tick_pixels[1] - x_tick_pixels[0]
        )
        values = -20.0 + (points[:, 1] - y_tick_pixels[0]) * 30.0 / (
            y_tick_pixels[1] - y_tick_pixels[0]
        )
        order = np.argsort(temperatures)
        rows.extend(
            (series, temperatures[index], values[index]) for index in order
        )

    plt.close(fig)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        output_path,
        np.asarray(rows, dtype=object),
        delimiter=",",
        fmt="%s",
        header="series,T_K,DeltaTad_K",
        comments="",
    )
    print(f"Saved {len(rows)} points to: {output_path}")


if __name__ == "__main__":
    main()
