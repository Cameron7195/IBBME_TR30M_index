"""IBBME_TR30M_index dataset."""
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import gzip
import json

# TODO(IBBME_TR30M_index): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(IBBME_TR30M_index): BibTeX citation
_CITATION = """
"""


class IbbmeTr30mIndex(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for IBBME_TR30M_index dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Remote hosted data.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(IBBME_TR30M_index): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            
            'promoterDataIdx': tfds.features.Scalar(dtype=tf.int32),
            'sampleID': tfds.features.Text(),
            'expression_TPM': tfds.features.Tensor(shape=(), dtype=tf.float32, encoding='ZLIB'),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=None,  # Set to `None` to disable
        homepage='https://bme.utoronto.ca//',
        disable_shuffling=False,
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(IBBME_TR30M): Downloads the data and defines the splits
    path = dl_manager.download_and_extract("https://archive.org/download/ibbme-tr3b-data/IBBME_TR3B_data.zip")

    # TODO(IBBME_TR30M): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train': self._generate_examples(path),
    }

  def _generate_examples(self, path):
    """Yields examples."""
    # TODO(IBBME_TR30M_index): Yields (key, example) tuples from the dataset
    expressionArr = np.load(gzip.GzipFile(path / 'expressionArr.npy.gz', 'r'))
    
    sampleToCellType = json.loads(open(path / 'sampleToCellType.json', 'r').read())

    for i in range(182522):
        expArrIdx1 = 0
        for sampleId, celltype in sampleToCellType.items():
            if np.random.uniform() > 0.01:
                continue
            exampleId = str(i) + '_' + str(expArrIdx1) + '_' + sampleId
            exp = np.float32(expressionArr[i, expArrIdx1])
            yield exampleId, {
                  'promoterDataIdx': i,
                  'sampleID': sampleId,
                  'expression_TPM': exp
            }
            expArrIdx1 += 1
        #if i > 2:
        #    break

