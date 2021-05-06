import tensorflow as tf

def build_wide_model(feature_column_dict, inputs, wmodel_dir, inc_numeric=False, 
                    name="wmodel.h5", ckpt_name="wmodel_checkpoint.h5"):

    """
    Builds and returns a wide-only model
    """
    
    #Should we send numeric columns as well?
    if inc_numeric:
        wide_only_feature_columns = feature_column_dict["numeric"] + feature_column_dict["crossed"]
    else:
        wide_only_feature_columns = feature_column_dict["crossed"]

    wmodel_path = wmodel_dir/name
    wmodel_checkpoint_path = wmodel_dir/ckpt_name

    #Simple early stopping
    w_es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
    w_mc = tf.keras.callbacks.ModelCheckpoint(str(wmodel_checkpoint_path), monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

    #Build the Wide Only Model
    w_feature_layer = tf.keras.layers.DenseFeatures(wide_only_feature_columns, name="dense_feature_layer")(inputs)
    w_linear_layer = tf.keras.layers.Dense(10, activation="softmax", name="wide_output")(w_feature_layer)
    wmodel = tf.keras.Model(inputs = inputs, outputs=w_linear_layer)
    wmodel.compile(optimizer="adam", 
                loss="categorical_crossentropy", 
                metrics=['accuracy', tf.keras.metrics.AUC()])    
    
    return wmodel, wmodel_path

def build_deep_model(feature_column_dict, inputs, dmodel_dir, 
                    name="dmodel.h5", ckpt_name="dmodel_checkpoint.h5"):

    """
    Builds and returns a deep-only model
    """

    #Passed object is a list instead of a dictionary
    if isinstance(feature_column_dict, list):
        deep_only_feature_columns = feature_column_dict[:]
    else:
        deep_only_feature_columns = []
        for feature_list in feature_column_dict.values():
            deep_only_feature_columns.extend(feature_list)
    
    deep_only_hidden_units = [512, 256, 128]
    dmodel_path = dmodel_dir/name
    dmodel_checkpoint_path = dmodel_dir/ckpt_name

    #simple early stopping
    d_es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
    d_mc = tf.keras.callbacks.ModelCheckpoint(str(dmodel_checkpoint_path), monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

    #Build the Wide Only Model
    d_feature_layer = tf.keras.layers.DenseFeatures(deep_only_feature_columns, name="dense_feature_layer")(inputs)
    d_hidden_layer_1 = tf.keras.layers.Dense(deep_only_hidden_units[0], activation="relu", name="fc_1")(d_feature_layer)
    d_hidden_layer_2 = tf.keras.layers.Dense(deep_only_hidden_units[1], activation="relu", name="fc_2")(d_hidden_layer_1)
    d_hidden_layer_3 = tf.keras.layers.Dense(deep_only_hidden_units[2], activation="relu", name="fc_3")(d_hidden_layer_2)
    d_output_layer = tf.keras.layers.Dense(10, activation="softmax", name="deep_output")(d_hidden_layer_3)

    dmodel = tf.keras.Model(inputs = inputs, outputs=d_output_layer)
    dmodel.compile(optimizer="adam", 
                loss="categorical_crossentropy", 
                metrics=['accuracy', tf.keras.metrics.AUC()])
    return dmodel, dmodel_path

def build_wide_and_deep_model(feature_column_dict, inputs, wdmodel_dir, 
                    name="wdmodel.h5", ckpt_name="wdmodel_checkpoint.h5"):

    """
    Builds and returns a wide & deep model
    """
    
    wide_wd_feature_columns = feature_column_dict["crossed"]
    deep_wd_feature_columns = feature_column_dict["numeric"] + feature_column_dict["embedding"]

    deep_wd_hidden_units = [512, 256, 128]

    wdmodel_path = wdmodel_dir/name
    wdmodel_checkpoint_path = wdmodel_dir/ckpt_name    

    #simple early stopping
    wd_es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
    wd_mc = tf.keras.callbacks.ModelCheckpoint(str(wdmodel_checkpoint_path), monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

    #Build the Wide Model
    w_feature_layer = tf.keras.layers.DenseFeatures(wide_wd_feature_columns, name="wide_feature_layer")(inputs)

    #Build the Deep Model
    d_feature_layer = tf.keras.layers.DenseFeatures(deep_wd_feature_columns, name="deep_feature_layer")(inputs)
    d_hidden_layer_1 = tf.keras.layers.Dense(deep_wd_hidden_units[0], activation="relu", name="deep_fc_1")(d_feature_layer)
    d_hidden_layer_2 = tf.keras.layers.Dense(deep_wd_hidden_units[1], activation="relu", name="deep_fc_2")(d_hidden_layer_1)
    d_hidden_layer_3 = tf.keras.layers.Dense(deep_wd_hidden_units[2], activation="relu", name="deep_fc_3")(d_hidden_layer_2)

    #Combine the Wide & Deep
    wd_both = tf.keras.layers.concatenate([w_feature_layer, d_hidden_layer_3])
    wd_output_layer = tf.keras.layers.Dense(10, activation="softmax", name="deep_output")(wd_both)
    wd_model = tf.keras.Model(inputs = inputs, outputs=wd_output_layer)
    wd_model.compile(optimizer="adam", 
                loss="categorical_crossentropy", 
                metrics=['accuracy', tf.keras.metrics.AUC()])
    
    return wd_model, wdmodel_path


def build_bayesian_model(feature_column_dict, inputs, wdmodel_dir, p=0.3, 
                    name="bmodel.h5", ckpt_name="bmodel_checkpoint.h5"):

    """
    Builds and returns a wide & deep bayesian bandit model via MC dropout
    """

    bayesian_path = bayesian_dir/name
    bayesian_checkpoint_path = bayesian_dir/ckpt_name

    bayesian_wide_feature_columns = feature_column_dict["crossed"]
    bayesian_deep_feature_columns = feature_column_dict["numeric"] + feature_column_dict["embedding"]
    bayesian_deep_hidden_units = [512, 256, 128]

    #simple early stopping
    bayesian_es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    bayesian_mc = tf.keras.callbacks.ModelCheckpoint(str(bayesian_checkpoint_path), monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

    #Build the Wide Model
    w_feature_layer = tf.keras.layers.DenseFeatures(bayesian_wide_feature_columns, name="wide_feature_layer")(inputs)

    #Build the Deep Model
    d_feature_layer = tf.keras.layers.DenseFeatures(bayesian_deep_feature_columns, name="deep_feature_layer")(inputs)
    d_hidden_layer_1 = tf.keras.layers.Dense(bayesian_deep_hidden_units[0], activation="relu", name="deep_fc_1")(d_feature_layer)
    d_hidden_layer_2 = tf.keras.layers.Dense(bayesian_deep_hidden_units[1], activation="relu", name="deep_fc_2")(d_hidden_layer_1)
    d_hidden_layer_3 = tf.keras.layers.Dense(bayesian_deep_hidden_units[2], activation="relu", name="deep_fc_3")(d_hidden_layer_2)

    #Combine the Wide & Deep
    bayesian_both = tf.keras.layers.concatenate([w_feature_layer, d_hidden_layer_3], name="concatenate") #Name is concatenate_1 for this

    #bayesian_pre_multihead_dropout = tf.keras.layers.Dropout(0.3)(bayesian_both, training=True) - Commented out to solve OOM issues
    bayesian_multihead = tf.keras.layers.Dense(64, activation="relu", name="multihead")(bayesian_both)
    bayesian_dropout = tf.keras.layers.Dropout(0.3, name="dropout")(bayesian_multihead, training=True)
    bayesian_output_layer = tf.keras.layers.Dense(10, activation="softmax", name="bayesian_model_output")(bayesian_dropout)
    bayesian_model = tf.keras.Model(inputs = inputs, outputs=bayesian_output_layer)
    bayesian_model.compile(optimizer="adam", 
                loss="categorical_crossentropy", 
                metrics=['accuracy', tf.keras.metrics.AUC()])
    
    return bayesian_model, bayesian_path