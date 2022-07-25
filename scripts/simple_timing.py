import argparse

from tt_sketch.sketch import hmt_sketch, stream_sketch
from tt_sketch.tensor import TensorTrain


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str)
    args = parser.parse_args()
    method_name = args.method
    if args.method not in ("HMT", "STTA"):
        print(f"Method '{method_name}' not supported")
        quit()

    shape = (50,) * 10
    tt_rank = 200
    tensor = TensorTrain.random(shape, rank=tt_rank)
    if args.method == "HMT":
        hmt_sketch(tensor, rank=tt_rank)
    elif args.method == "STTA":
        stream_sketch(tensor, left_rank=tt_rank, right_rank=tt_rank * 2)
