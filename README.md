# pix2app AI 写程序
- 通常由设计稿转化为网页的过程由前端工程师来实现，我们的演示展示了由AI来完成简单页面的制作，它通过“观察”图片，输出编码结果。
- 输入图片基于腾讯WeUI design设计规范，输出结果可以是H5页面/微信小程序/或手机App。
- 前端工程师可以解放一部分用于初级布局的工作时间，更专注于业务流程，后台交互部分。
- 仍有大量问题等待解决中......

![](https://raw.githubusercontent.com/pix2app/pix2app/master/demo.gif)

## 用法

	$ cd pix2app
  
下载训练数据

	$ wget https://s3-ap-northeast-1.amazonaws.com/pix2app/train_data.tar.gz
	$ tar -xzf train_data.tar.gz
  
开始训练

	$ python pix2app.py
  
测试

	$ python test.py imgs/0.png
