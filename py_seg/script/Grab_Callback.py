# -- coding: utf-8 --

import pyads
import sys
import  readchar
from ctypes import *
from deeplab import DeeplabV3
from PIL import Image


libc = CDLL("/lib/aarch64-linux-gnu/libc.so.6")

import cv2
import numpy as np

sys.path.append("./MvImport")

from MvImport.MvCameraControl_class import *
winfun_ctype = CFUNCTYPE

stFrameInfo = POINTER(MV_FRAME_OUT_INFO_EX)
pData = POINTER(c_ubyte)
FrameInfoCallBack = winfun_ctype(None, pData, stFrameInfo, c_void_p)

deeplab = DeeplabV3()
name_classes    = ["background", "weld"]

# ads连接
#ads_route = pyads.Connection('192.168.1.2.1.1', 851)
#ads_route.open()


# 图像回调
def image_callback(pData, pFrameInfo, pUser):
	global img_buff
	img_buff = None
	stFrameInfo = cast(pFrameInfo, POINTER(MV_FRAME_OUT_INFO_EX)).contents
	if stFrameInfo:
		print("get one frame: Width[%d], Height[%d], nFrameNum[%d]" % (
		stFrameInfo.nWidth, stFrameInfo.nHeight, stFrameInfo.nFrameNum))
		stConvertParam = MV_CC_PIXEL_CONVERT_PARAM()
		memset(byref(stConvertParam), 0, sizeof(stConvertParam))

		if IsImageColor(stFrameInfo.enPixelType) == 'mono':
			stConvertParam.enDstPixelType = PixelType_Gvsp_Mono8
			nConvertSize = stFrameInfo.nWidth * stFrameInfo.nHeight
		else:
			print("not support!!!")

	if img_buff is None:
		img_buff = (c_ubyte * stFrameInfo.nFrameLen)()
	stConvertParam.nWidth = stFrameInfo.nWidth
	stConvertParam.nHeight = stFrameInfo.nHeight
	stConvertParam.pSrcData = cast(pData, POINTER(c_ubyte))
	stConvertParam.nSrcDataLen = stFrameInfo.nFrameLen
	stConvertParam.enSrcPixelType = stFrameInfo.enPixelType
	stConvertParam.pDstBuffer = (c_ubyte * nConvertSize)()
	stConvertParam.nDstBufferSize = nConvertSize
	ret = cam.MV_CC_ConvertPixelType(stConvertParam)
	if ret != 0:
		print("convert pixel fail! ret[0x%x]" % ret)
		del stConvertParam.pSrcData
		sys.exit()
	else:
		# 转OpenCV 黑白处理
		if IsImageColor(stFrameInfo.enPixelType) == 'mono':
			img_buff = (c_ubyte * stConvertParam.nDstLen)()
			libc.memcpy(byref(img_buff), stConvertParam.pDstBuffer, stConvertParam.nDstLen)
			img_buff = np.frombuffer(img_buff, count=int(stConvertParam.nDstLen), dtype=np.uint8)
			img_buff = img_buff.reshape((stFrameInfo.nHeight, stFrameInfo.nWidth))
			image = Image.fromarray(np.uint8(img_buff))
			
      # semantic segmentation
			blend_image, r_image = deeplab.detect_image(image, name_classes=name_classes)
			r_image = caculate_area(r_image)
			# show_image = np.hstack((np.array(blend_image), np.array(r_image)))
			print("have predicted picture no.%d" % stFrameInfo.nFrameNum)
			image_show( blend_image, r_image)  # 显示图像函数

CALL_BACK_FUN = FrameInfoCallBack(image_callback)

# 计算预测区域
def caculate_area(image):
	image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
	# cv2.imwrite("test1.jpg", image)
	ret, image = cv2.threshold(image, 10, 255, cv2.THRESH_BINARY)
	image = cv2.dilate(image, (41,41))
	# image = cv2.morphologyEx(image, cv2.MORPH_OPEN, (5, 5))
	# cv2.imwrite("test2.jpg", image)
	contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL,  cv2.CHAIN_APPROX_SIMPLE)
	contours_list = [ ]
	for c in contours:
		M = cv2.moments(c)
		cx = int(M["m10"] / M["m00"])
		cy = int(M["m01"] / M["m00"])
		contours_list.append((cx, cy, M["m00"]))
	contours_list.sort(key = lambda x:x[2], reverse=True)
	# print("list:   ",contours_list)
	center_point = contours_list[0] 
	cv2.drawMarker(image, (center_point[0], center_point[1]), (0,0,0), cv2.MARKER_CROSS, 30, 5 )
	pix_coordination = "pix: ( %d, %d )" % (center_point[0], center_point[1])
	x_world, y_world = pix2world(center_point[0], center_point[1], 1800, 1800, 680, 540, 0.277)
	world_coordination = "world: ( %f, %f )" % (x_world, y_world)
	cv2.putText(image, pix_coordination,  (center_point[0] - 80, center_point[1] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))
	cv2.putText(image, world_coordination,  (center_point[0] - 80, center_point[1] + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))
	# ads_pub(x_world, y_world)
	return image

# 计算是世界坐标位置
def pix2world(x, y, fx, fy, cx, cy, z):
	x_world = (x - cx) / fx * z
	y_world = (y - cy) / fy * z
	return x_world, y_world

# ads发布信息
def ads_pub(x,y):
	global ads_route
	x_pub = ads_route.write_by_name("MAIN.", x, pyads.PLCTYPE_LREAL)
	y_pub = ads_route.write_by_name("MAIN.", y, pyads.PLCTYPE_LREAL)

# 显示图像
def image_show(image1, image2):
    image1 = cv2.resize(np.array(image1), (1280, 1024), interpolation=cv2.INTER_AREA)
    image2 = cv2.resize(np.array(image2), (1280, 1024), interpolation=cv2.INTER_AREA)
    cv2.namedWindow('test1', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('test1', image1)
    cv2.namedWindow('test2', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('test2', image2)
    k = cv2.waitKey(1) & 0xff

# 判读图像格式是彩色还是黑白
def IsImageColor(enType):
    dates = {
        PixelType_Gvsp_RGB8_Packed: 'color',
        PixelType_Gvsp_BGR8_Packed: 'color',
        PixelType_Gvsp_YUV422_Packed: 'color',
        PixelType_Gvsp_YUV422_YUYV_Packed: 'color',
        PixelType_Gvsp_BayerGR8: 'color',
        PixelType_Gvsp_BayerRG8: 'color',
        PixelType_Gvsp_BayerGB8: 'color',
        PixelType_Gvsp_BayerBG8: 'color',
        PixelType_Gvsp_BayerGB10: 'color',
        PixelType_Gvsp_BayerGB10_Packed: 'color',
        PixelType_Gvsp_BayerBG10: 'color',
        PixelType_Gvsp_BayerBG10_Packed: 'color',
        PixelType_Gvsp_BayerRG10: 'color',
        PixelType_Gvsp_BayerRG10_Packed: 'color',
        PixelType_Gvsp_BayerGR10: 'color',
        PixelType_Gvsp_BayerGR10_Packed: 'color',
        PixelType_Gvsp_BayerGB12: 'color',
        PixelType_Gvsp_BayerGB12_Packed: 'color',
        PixelType_Gvsp_BayerBG12: 'color',
        PixelType_Gvsp_BayerBG12_Packed: 'color',
        PixelType_Gvsp_BayerRG12: 'color',
        PixelType_Gvsp_BayerRG12_Packed: 'color',
        PixelType_Gvsp_BayerGR12: 'color',
        PixelType_Gvsp_BayerGR12_Packed: 'color',
        PixelType_Gvsp_Mono8: 'mono',
        PixelType_Gvsp_Mono10: 'mono',
        PixelType_Gvsp_Mono10_Packed: 'mono',
        PixelType_Gvsp_Mono12: 'mono',
        PixelType_Gvsp_Mono12_Packed: 'mono'}
    return dates.get(enType, '未知')

# 阻塞退出
def press_q_exit():
	while True:
		key = readchar.readkey()
		if (key == 'q'):
			print("system shut down!")
			break

if __name__ == "__main__":

	SDKVersion = MvCamera.MV_CC_GetSDKVersion()
	print ("SDKVersion[0x%x]" % SDKVersion)

	deviceList = MV_CC_DEVICE_INFO_LIST()
	tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
	
	# ch:枚举设备 | en:Enum device
	ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
	if ret != 0:
		print ("enum devices fail! ret[0x%x]" % ret)
		sys.exit()
	
	if deviceList.nDeviceNum == 0:
		print ("find no device!")
		sys.exit()

	print ("find %d devices!" % deviceList.nDeviceNum)

	for i in range(0, deviceList.nDeviceNum):
		mvcc_dev_info = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
		if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
			print ("\ngige device: [%d]" % i)
			strModeName = ""
			for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chModelName:
				strModeName = strModeName + chr(per)
			print ("device model name: %s" % strModeName)

			nip1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24)
			nip2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16)
			nip3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8)
			nip4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff)
			print ("current ip: %d.%d.%d.%d\n" % (nip1, nip2, nip3, nip4))
		elif mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
			print ("\nu3v device: [%d]" % i)
			strModeName = ""
			for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chModelName:
				if per == 0:
					break
				strModeName = strModeName + chr(per)
			print ("device model name: %s" % strModeName)

			strSerialNumber = ""
			for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber:
				if per == 0:
					break
				strSerialNumber = strSerialNumber + chr(per)
			print ("user serial number: %s" % strSerialNumber)

	nConnectionNum = 0

	if int(nConnectionNum) >= deviceList.nDeviceNum:
		print ("intput error!")
		sys.exit()
	
	# ch:创建相机实例 | en:Creat Camera Object
	cam = MvCamera()

	# ch:选择设备并创建句柄 | en:Select device and create handle
	stDeviceList = cast(deviceList.pDeviceInfo[int(nConnectionNum)], POINTER(MV_CC_DEVICE_INFO)).contents

	ret = cam.MV_CC_CreateHandle(stDeviceList)
	if ret != 0:
		print ("create handle fail! ret[0x%x]" % ret)
		sys.exit()

	# ch:打开设备 | en:Open device
	ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
	if ret != 0:
		print ("open device fail! ret[0x%x]" % ret)
		sys.exit()
	
	# ch:探测网络最佳包大小(只对GigE相机有效) | en:Detection network optimal package size(It only works for the GigE camera)
	if stDeviceList.nTLayerType == MV_GIGE_DEVICE:
		nPacketSize = cam.MV_CC_GetOptimalPacketSize()
		if int(nPacketSize) > 0:
			ret = cam.MV_CC_SetIntValue("GevSCPSPacketSize",nPacketSize)
			if ret != 0:
				print ("Warning: Set Packet Size fail! ret[0x%x]" % ret)
		else:
			print ("Warning: Get Packet Size fail! ret[0x%x]" % nPacketSize)

	# ch:设置触发模式为off | en:Set trigger mode as off
	ret = cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
	if ret != 0:
		print ("set trigger mode fail! ret[0x%x]" % ret)
		sys.exit()

	# ch:注册抓图回调 | en:Register image callback
	ret = cam.MV_CC_RegisterImageCallBackEx(CALL_BACK_FUN,None)
	if ret != 0:
		print ("register image callback fail! ret[0x%x]" % ret)
		sys.exit()

	#设置相机属性
	cam.MV_CC_SetIntValue("Width", 1280)
	cam.MV_CC_SetIntValue("Height", 1024)
	cam.MV_CC_SetIntValue("OffsetX", 320)
	cam.MV_CC_SetIntValue("OffsetY", 88)
	cam.MV_CC_SetEnumValue("TriggerMode", 0)
	cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", True)
	cam.MV_CC_SetFloatValue("AcquisitionFrameRate",5)
	cam.MV_CC_SetFloatValue("ExposureTime", 15000)
	cam.MV_CC_SetEnumValue("GainAuto", 0)
	# cam.MV_CC_SetFloatValue("Gamma", 0.7)
	# print(cam.MV_CC_GetFloatValue("ResultingFrameRate"))

	# ch:开始取流 | en:Start grab image
	ret = cam.MV_CC_StartGrabbing()
	if ret != 0:
		print ("start grabbing fail! ret[0x%x]" % ret)
		sys.exit()

	press_q_exit()

	# ch:停止取流 | en:Stop grab image
	ret = cam.MV_CC_StopGrabbing()
	if ret != 0:
		print ("stop grabbing fail! ret[0x%x]" % ret)
		sys.exit()

	# ch:关闭设备 | Close device
	ret = cam.MV_CC_CloseDevice()
	if ret != 0:
		print ("close deivce fail! ret[0x%x]" % ret)
		sys.exit()

	# ch:销毁句柄 | Destroy handle
	ret = cam.MV_CC_DestroyHandle()
	if ret != 0:
		print ("destroy handle fail! ret[0x%x]" % ret)
		sys.exit()
