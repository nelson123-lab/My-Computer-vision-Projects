from tf_explain.callbacks.occlusion_sensitivity import OcclusionSensitivityCallback
import datetime
%load_ext tensorboard
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
o_callbacks = [OcclusionSensitivityCallback(validation_data=(vis_test, vis_lab),class_index=2,patch_size=4),]
model_TF.compile(optimizer=keras.optimizers.Adam(lr=0.001), loss='binary_crossentropy', metrics=[fbeta])
model_TF.fit(vis_test, vis_lab, epochs=10, verbose=1, callbacks=[tensorboard_callback, o_callbacks])
