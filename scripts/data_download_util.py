import urllib.request
import os.path
import os
from tqdm import tqdm

DATA_FOLDER = "data"


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(
        unit="B",
        unit_scale=True,
        miniters=1,
        desc="downloading " + url.split("/")[-1],
    ) as t:
        urllib.request.urlretrieve(
            url, filename=output_path, reporthook=t.update_to
        )


def download_data(url):
    try:
        os.mkdir(DATA_FOLDER)
    except FileExistsError:
        pass

    filename = url.split("/")[-1]
    filepath = os.path.join(DATA_FOLDER, filename)
    if os.path.isfile(filepath):
        print("File already downloaded")
    else:
        download_url(url, filepath)
