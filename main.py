import datetime
import os
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import LSTM


from models import Transformer, LSTMM

if __name__=='__main__':
  # fetch data
  zip_path = tf.keras.utils.get_file(
      origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
      fname='jena_climate_2009_2016.csv.zip',
      extract=True)
  csv_path, _ = os.path.splitext(zip_path)

  df = pd.read_csv(csv_path)
  # slice [start:stop:step], starting from index 5 take every 6th record.
  #df = df[5::6]

  date_time = pd.to_datetime(df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')

  #print(df[df["T (degC)"] < 0]["T (degC)"])
  df = df.drop(["max. wv (m/s)", "wv (m/s)"], axis=1)

  day = 24*60*60
  year = (365.2425)*day

  timestamp_s = date_time.map(datetime.datetime.timestamp)
  df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
  df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
  df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
  df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))


  print(df.head(20))



  # モデルのインスタンスを作成
  model = Transformer()
  #model = LSTMM()
  loss_object = tf.keras.losses.BinaryCrossentropy()
  optimizer = tf.keras.optimizers.Adam(0.01)
  #
  train_loss = tf.keras.metrics.Mean(name='train_loss')
  train_accuracy = tf.keras.metrics.BinaryAccuracy()

  test_loss = tf.keras.metrics.Mean(name='test_loss')
  test_accuracy = tf.keras.metrics.BinaryAccuracy()

  @tf.function
  def train_step(seq, labels):
    with tf.GradientTape() as tape:
      predictions = model(seq)
      tf.print(predictions[0])
      loss = loss_object(labels, predictions)
    print(model.trainable_variables)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

  @tf.function
  def test_step(seq, labels):
    predictions = model(seq)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)

  column_indices = {name: i for i, name in enumerate(df.columns)}

  n = len(df)
  train_df = df[0:int(n*0.7)]
  val_df = df[int(n*0.7):int(n*0.9)]
  test_df = df[int(n*0.9):]

  num_features = df.shape[1]

  train_mean = train_df.mean()
  train_std = train_df.std()

  train_df = (train_df - train_mean) / train_std
  val_df = (val_df - train_mean) / train_std
  test_df = (test_df - train_mean) / train_std

  print(train_df.describe())


  def get_median(v):
     v = tf.reshape(v, [-1])
     m = v.get_shape()[0]//2
     return tf.nn.top_k(v, m).values[m-1]

  print(get_median(train_df["T (degC)"]))
  print(train_df.head(20))

  x = 0.009
  train_df["T (degC)"][train_df["T (degC)"] > x] = 1
  test_df["T (degC)"][test_df["T (degC)"] > x] = 1
  val_df["T (degC)"][val_df["T (degC)"] > x] = 1

  train_df["T (degC)"][train_df["T (degC)"] <= x] = 0
  test_df["T (degC)"][test_df["T (degC)"] <= x] = 0
  val_df["T (degC)"][val_df["T (degC)"] <= x] = 0

  print(sum(train_df["T (degC)"]))
  print(train_df["T (degC)"].shape)

  class WindowGenerator():
    def __init__(self, input_width, label_width, shift,
                 train_df=train_df, val_df=val_df, test_df=test_df,
                 label_columns=None):
      # Store the raw data.
      self.train_df = train_df
      self.val_df = val_df
      self.test_df = test_df

      # Work out the label column indices.
      self.label_columns = label_columns
      if label_columns is not None:
        self.label_columns_indices = {name: i for i, name in
                                      enumerate(label_columns)}
      self.column_indices = {name: i for i, name in
                             enumerate(train_df.columns)}

      # Work out the window parameters.
      self.input_width = input_width
      self.label_width = label_width
      self.shift = shift

      self.total_window_size = input_width + shift

      self.input_slice = slice(0, input_width)
      self.input_indices = np.arange(self.total_window_size)[self.input_slice]

      self.label_start = self.total_window_size - self.label_width
      self.labels_slice = slice(self.label_start, None)
      self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
      return '\n'.join([
          f'Total window size: {self.total_window_size}',
          f'Input indices: {self.input_indices}',
          f'Label indices: {self.label_indices}',
          f'Label column name(s): {self.label_columns}'])

    def split_window(self, features):
      inputs = features[:, self.input_slice, :]
      labels = features[:, self.labels_slice, :]
      if self.label_columns is not None:
        labels = tf.stack(
            [labels[:, :, self.column_indices[name]] for name in self.label_columns],
            axis=-1)

      # Slicing doesn't preserve static shape information, so set the shapes
      # manually. This way the `tf.data.Datasets` are easier to inspect.
      inputs.set_shape([None, self.input_width, None])
      labels.set_shape([None, self.label_width, None])

      return inputs, labels

    def make_dataset(self, data):
      data = np.array(data, dtype=np.float32)
      ds = tf.keras.preprocessing.timeseries_dataset_from_array(
          data=data,
          targets=None,
          sequence_length=self.total_window_size,
          sequence_stride=1,
          shuffle=True,
          batch_size=32,)

      ds = ds.map(self.split_window)
      return ds

    @property
    def train(self):
      return self.make_dataset(self.train_df)

    @property
    def val(self):
      return self.make_dataset(self.val_df)

    @property
    def test(self):
      return self.make_dataset(self.test_df)

    @property
    def example(self):
      """Get and cache an example batch of `inputs, labels` for plotting."""
      result = getattr(self, '_example', None)
      if result is None:
        # No example batch was found, so get one from the `.train` dataset
        result = next(iter(self.train))
        # And cache it for next time
        self._example = result
      return result

  w2 = WindowGenerator(input_width=9, label_width=1, shift=1,
                       train_df=train_df, test_df=test_df,val_df=val_df,
                     label_columns=['T (degC)'])


  EPOCHS = 100

  for epoch in range(EPOCHS):
    for images, labels in w2.train.take(100):
      train_step(images, labels)

    for test_images, test_labels in w2.test.take(10):
      test_step(test_images, test_labels)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print (template.format(epoch+1,
                           train_loss.result(),
                           train_accuracy.result()*100,
                           test_loss.result(),
                           test_accuracy.result()*100))

    # 次のエポック用にメトリクスをリセット
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

