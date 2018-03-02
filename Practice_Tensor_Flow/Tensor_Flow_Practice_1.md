# Tensor Flow Training Practice :

+ Generally It is difficult to practice Training
+ The more you type The more practice you have
+ This is going through the steps for Training a LSTM stack to do a Sentiment RNN analysis
+ Before this there is a part where you create your training , test and validation sets
+ Tensor Flow : https://www.tensorflow.org/api_docs/python/tf/contrib/rnn

## Building the Graph

```
graph = tf.Graph()
with graph.as_default():
	inputs_ = tf.placeholder(tf.int32, [None, None], name='inputs')
	labels_ = tf.placeholder(tf.int32, [None, None], name='labels')
	keep_prob = tf.placeholder(tf.float32, name='keep_prob')
```

## Embedding Layer
  + As it is not possible to do a one-hot encoding all the time for each of the words
  + embedding is the loop up matrix

```
embed_size=300
with graph.as_default():
	embedding=tf.Variable(tf.random_uniform((n_words, embed_size),-1,1))
	embed = tf.nn.embedding_lookup(embedding, inputs_)
```

## LSTM Cell for stacking up the LSTM Layers

```
with graph.as_default():
	lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
	drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep=keep_prob)
	cell = tf.contrib.rnn.MultiRNNCell([drop]*lstm_layers)
	initial_state = cell.zero_state(batch_size, tf.float32)
```

## Forward Pass through RNN

```
with graph.as_default():
	outputs, final_state = tf.nn.dynamic_rnn(cell, inputs=embed, initial_state=initial_state)
```

## Output
 + For this case we will be getting the output from the last node ie, outputs[:, -1]
```
with graph.as_default():
	predictions = tf.contrib.layers.fully_connected(outputs[:,-1], 1, activation_fn=tf.sigmoid)
	cost = tf.losses.mean_squared_error(labels_, predictions)
	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
```

# Validation accuracy
 + To calculate the accuracy during the validation pass
```
with graph.as_default():
	correct_pred = tf.equal(tf.cast(tf.round(predictions),tf.int32), labels_)
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
```
# Batching
 + Returns arrays with sizes equal to [batch_size]
```
def get_batches(x, y, batch_size=100):
	n_batches = len(x)//100
	x,y = x[:n_batches*batch_size], y[:n_batches*batch_size]
	for ii in range(0, len(x), batch_size):
		yield x[ii:ii+batch_size], y[ii:ii+batch_size]
```
## Training the network

```
epochs = 10
with graph.as_default():
	saver = tf.train.Saver()

with tf.Session(graph=graph) as sess:
	sess.run(tf.global_variables_initializer())
	iter=1
	for e in range(epochs):
		state = sess.run(initial_state)

		for ii, (x,y) in enumerate(get_batches(train_x, train_y, batch_size),1):
			feed = {
				inputs_:x,
				labels_:y[:, None],
				keep_prob:0.5,
				initial_state: state}
			loss,state, _ = sess.run([cost, final_state, optimizer], feed_dict=feed)
			if iteration%5==0:
				print("Epoch: {}/{}".format(e, epochs), "Iteration: {}".format(iteration), "Train loss: {:.3f}".format(loss))
			if iteration%25==0:
				val_acc = []
				val_state  = sess.run(cell.zero_state(batch_size, tf.float32))
				for x,y in get_batches(val_x, val_y, batch_size):
					feed = {
						inputs_:x,
						labels_y:[:,None],
						keep_prob:1,
						initial_state:val_state}
					batch_acc, val_state = sess.run([accuracy, final_state], feed_dict=feed)
					val_acc.append(batch_acc)
				print("Validation acc : {:.3f}".format(np.mean(val_acc)))
				iteration+=1
	saver.save(sess, "checkpoints/sentiment.ckpt")
```
