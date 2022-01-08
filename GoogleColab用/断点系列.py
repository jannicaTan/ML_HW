#1.回调参数：
#可以是格式化的字符串，里面的占位符将会被epoch值和传入的监控指标所填入
checkpoint_filepath = '/content/drive/MyDrive/CIFAR10/checkpoint/Cifar10.{epoch:02d}-{val_loss:.4f}.H5'

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                       save_weights_only=True,
                                       verbose=0,
                                       save_freq='epoch'), 
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3),
]

#2.加载模型
checkpoint_dir = '/content/drive/MyDrive/CIFAR10/checkpoint/'#检查点文件保存目录
model_filename=tf.train.latest_checkpoint(checkpoint_dir)
#得到最新的检查点文件
if model_filename != None:
  model.load_weights(model_filename)
  print("已成功加载".format(model_filename))
else:
  print("没有此文件")


#3.断点续训：加载模型
model_filename = '/content/drive/MyDrive/CIFAR10/cifarCNNModel.h5'
try:
  model.load_weights(model_filename)
  print("已成功加载")
except:
  print("没有此文件")

#4.训练模型
train_history = model.fit(x_train, y_train,
                          validation_split = 0.2,
                          epochs = train_epochs,
                          batch_size = batch_size,
                          callbacks = callbacks, # 加了回调参数
                          verbose = 2)

#5.保存模型
model_filename = '/content/drive/MyDrive/CIFAR10/cifarCNNModel.h5'
model.save_weights(model_filename)#保存模型参数，不会保存模型结构
print("已经保存模型权重! ")