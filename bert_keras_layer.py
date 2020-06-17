from bert_tokenizer import *
import os
import bert
import tensorflow as tf

max_seq_length = 128
classes = 10

def create_bert_layer(model_dir):
    bert_params = bert.params_from_pretrained_ckpt(model_dir)
    bert_layer = bert.BertModelLayer.from_params(bert_params, name="bert")
    bert_layer.apply_adapter_freeze()
    checkpoint_name = os.path.join(model_dir, "bert_model.ckpt.data-00000-of-00001")
    return bert_layer

if __name__ == '__main__':
    tokenizer = create_tokenizer('uncased_L-12_H-768_A-12')

    df = get_filtered_nyt_data('nyt_data_from_2015.pkl')
    df['category_labels'] = df['section'].astype('category').cat.codes
    train_df, test_df = train_test_split(df, random_state=42)
    train_ids, train_labels = tokenize_data(train_df['abstract'], train_df['category_labels'], tokenizer)
    test_ids, test_labels = tokenize_data(df['abstract'], train_df['category_labels'], tokenizer)
    bert_layer = create_bert_layer('uncased_L-12_H-768_A-12')

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(max_seq_length,), dtype='int32', name='input_ids'),
        bert_layer,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(classes, activation=tf.nn.softmax)
    ])

    model.build(input_shape=(None, max_seq_length))

    model.compile(loss='categorical_crossentropy', optimizer=tf.optimizers.Adam(lr=0.00001), metrics=['accuracy'])
    print(model.summary())

    model.fit(train_ids, train_labels, validation_data=(test_ids, test_labels), epochs=5)
