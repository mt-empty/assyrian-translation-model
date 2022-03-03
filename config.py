import datasets

_DESCRIPTION = """Assyrian to english dataset
"""
_HOMEPAGE_URL = "http://www.deepneuron.com"

_VERSION = "0.0.1"

_LANGUAGE_PAIRS = [
    ("en", "as"),
]

# The code is based on the KDE4 dataset script found on the datasets github repo:
# https://github.com/huggingface/datasets/blob/master/datasets/kde4/kde4.py
# the documentation for the following classes can be found here:
# https://huggingface.co/docs/datasets/add_dataset.html


class EnSaConfig(datasets.BuilderConfig):
    def __init__(self, *args, lang1=None, lang2=None, **kwargs):
        super().__init__(
            *args,
            name=f"{lang1}-{lang2}",
            **kwargs,
        )
        self.lang1 = lang1
        self.lang2 = lang2


class EnSa(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        EnSaConfig(
            lang1=lang1,
            lang2=lang2,
            description=f"Translating {lang1} to {lang2} or vice versa",
            version=datasets.Version(_VERSION),
        )
        for lang1, lang2 in _LANGUAGE_PAIRS
    ]
    BUILDER_CONFIG_CLASS = EnSaConfig

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "translation": datasets.Translation(
                        languages=(self.config.lang1, self.config.lang2)
                    ),
                },
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE_URL,
        )

    def _split_generators(self, dl_manager):

        path = dl_manager.download_and_extract(self.config.data_files)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"datapath": path},
            )
        ]

    def _generate_examples(self, datapath):
        l1, l2 = self.config.lang1, self.config.lang2
        l1_path = datapath.get("train")[0]
        l2_path = datapath.get("train")[1]
        with open(l1_path, encoding="utf-8") as f1, open(
            l2_path, encoding="utf-8"
        ) as f2:
            for sentence_counter, (x, y) in enumerate(zip(f1, f2)):
                x = x.strip()
                y = y.strip()
                result = (
                    sentence_counter,
                    {
                        "id": str(sentence_counter),
                        "translation": {l1: x, l2: y},
                    },
                )
                yield result
