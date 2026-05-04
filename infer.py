# import cv2
# import torch
# import torch.nn.functional as F
# from torchvision import transforms
# from models.MobileNetV2 import MobileNetV2
# import numpy as np
#
#
# class HandClassifier:
#     def __init__(self, model_path, num_classes=5, device='cuda'):
#         """
#         手型分类器
#         Args:
#             model_path: 模型权重路径
#             num_classes: 类别数量
#             device: 运行设备
#         """
#         self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
#         self.num_classes = num_classes
#
#         # 加载模型
#         self.model = MobileNetV2(num_classes=num_classes).to(self.device)
#         self.model.load_state_dict(torch.load(model_path, map_location=self.device))
#         self.model.eval()
#
#         # 预处理
#         self.transform = transforms.Compose([
#             transforms.ToPILImage(),
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])
#         ])
#
#         # 类别名称（根据你的实际类别修改）
#         self.class_names = {
#             0: "手型0",
#             1: "手型1",
#             2: "手型2",
#             3: "手型3",
#             4: "手型4"
#         }
#
#     def preprocess(self, image):
#         """
#         预处理图像
#         Args:
#             image: BGR格式的numpy数组 (H, W, 3)
#         Returns:
#             预处理后的tensor (1, 3, 224, 224)
#         """
#         # BGR转RGB
#         image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         # 应用预处理
#         tensor = self.transform(image_rgb)
#         # 添加batch维度
#         tensor = tensor.unsqueeze(0)
#         return tensor
#
#     def predict(self, image):
#         """
#         预测单张图像
#         Args:
#             image: BGR格式的numpy数组
#         Returns:
#             class_id: 预测的类别ID
#             confidence: 置信度
#             all_probs: 所有类别的概率
#         """
#         # 预处理
#         tensor = self.preprocess(image)
#         tensor = tensor.to(self.device)
#
#         # 推理
#         with torch.no_grad():
#             output = self.model(tensor)
#             probs = F.softmax(output, dim=1)
#             confidence, pred = torch.max(probs, 1)
#
#         class_id = pred.item()
#         confidence = confidence.item()
#         all_probs = probs.cpu().numpy()[0]
#
#         return class_id, confidence, all_probs
#
#     def predict_batch(self, images):
#         """
#         批量预测
#         Args:
#             images: BGR格式的numpy数组列表
#         Returns:
#             class_ids: 预测的类别ID列表
#             confidences: 置信度列表
#         """
#         batch_tensors = []
#         for image in images:
#             tensor = self.preprocess(image)
#             batch_tensors.append(tensor)
#
#         batch = torch.cat(batch_tensors, dim=0).to(self.device)
#
#         with torch.no_grad():
#             output = self.model(batch)
#             probs = F.softmax(output, dim=1)
#             confidences, preds = torch.max(probs, 1)
#
#         return preds.cpu().numpy(), confidences.cpu().numpy()
#
#
# def main():
#     # 初始化分类器
#     model_path = "runs/train/20260422_083031/best_model.pth"  # 修改为你的模型路径
#     classifier = HandClassifier(model_path, num_classes=5)
#
#     # 打开摄像头
#     cap = cv2.VideoCapture(0)
#
#     # 设置摄像头参数
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#
#     print("按 'q' 退出")
#     print("按 's' 保存当前帧")
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("无法获取摄像头画面")
#             break
#
#         # 预处理并推理
#         class_id, confidence, all_probs = classifier.predict(frame)
#
#         # 显示结果
#         class_name = classifier.class_names[class_id]
#
#         # 在图像上绘制结果
#         # 显示类别和置信度
#         text = f"{class_name}: {confidence:.2%}"
#         cv2.putText(frame, text, (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#
#         # 显示所有类别的概率（可选）
#         y_offset = 60
#         for i in range(classifier.num_classes):
#             prob_text = f"Class {i}: {all_probs[i]:.2%}"
#             color = (0, 255, 0) if i == class_id else (255, 255, 255)
#             cv2.putText(frame, prob_text, (10, y_offset + i * 25),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
#
#         # 显示FPS（可选）
#         cv2.imshow('Hand Classifier', frame)
#
#         key = cv2.waitKey(1) & 0xFF
#         if key == ord('q'):
#             break
#         elif key == ord('s'):
#             # 保存当前帧
#             cv2.imwrite("saved_frame.jpg", frame)
#             print("已保存当前帧")
#
#     cap.release()
#     cv2.destroyAllWindows()
#
#
# # 单张图片推理示例
# def predict_single_image():
#     """对单张图片进行推理"""
#     classifier = HandClassifier("runs/train/20260422_000341/best_model.pth", num_classes=5)
#
#     # 读取图片
#     image = cv2.imread("test_image.jpg")
#     if image is None:
#         print("无法读取图片")
#         return
#
#     # 推理
#     class_id, confidence, all_probs = classifier.predict(image)
#
#     print(f"预测类别: {class_id}")
#     print(f"置信度: {confidence:.4f}")
#     print(f"所有类别概率: {all_probs}")
#
#     # 显示结果
#     class_name = classifier.class_names[class_id]
#     text = f"{class_name}: {confidence:.2%}"
#     cv2.putText(image, text, (10, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#     cv2.imshow("Result", image)
#     cv2.waitKey(0)
#
#
# if __name__ == "__main__":
#     # 摄像头实时推理
#     main()
#
#     # 或者单张图片推理
#     # predict_single_image()
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from models.MobileNetV2 import MobileNetV2
import numpy as np
from collections import deque
import time


class HandClassifier:
    def __init__(self, model_path, num_classes=5, device='cuda'):
        """
        手型分类器
        Args:
            model_path: 模型权重路径
            num_classes: 类别数量
            device: 运行设备
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes

        # 加载模型
        self.model = MobileNetV2(num_classes=num_classes).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # 预处理
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # 类别名称（根据你的实际类别修改）
        self.class_names = {
            0: "手型0",
            1: "手型1",
            2: "手型2",
            3: "手型3",
            4: "手型4"
        }

    def preprocess(self, image):
        """
        预处理图像
        Args:
            image: BGR格式的numpy数组 (H, W, 3)
        Returns:
            预处理后的tensor (1, 3, 224, 224)
        """
        # BGR转RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 应用预处理
        tensor = self.transform(image_rgb)
        # 添加batch维度
        tensor = tensor.unsqueeze(0)
        return tensor

    def predict(self, image):
        """
        预测单张图像
        Args:
            image: BGR格式的numpy数组
        Returns:
            class_id: 预测的类别ID
            confidence: 置信度
            all_probs: 所有类别的概率
        """
        # 预处理
        tensor = self.preprocess(image)
        tensor = tensor.to(self.device)

        # 推理
        with torch.no_grad():
            output = self.model(tensor)
            probs = F.softmax(output, dim=1)
            confidence, pred = torch.max(probs, 1)

        class_id = pred.item()
        confidence = confidence.item()
        all_probs = probs.cpu().numpy()[0]

        return class_id, confidence, all_probs

    def predict_batch(self, images):
        """
        批量预测
        Args:
            images: BGR格式的numpy数组列表
        Returns:
            class_ids: 预测的类别ID列表
            confidences: 置信度列表
        """
        batch_tensors = []
        for image in images:
            tensor = self.preprocess(image)
            batch_tensors.append(tensor)

        batch = torch.cat(batch_tensors, dim=0).to(self.device)

        with torch.no_grad():
            output = self.model(batch)
            probs = F.softmax(output, dim=1)
            confidences, preds = torch.max(probs, 1)

        return preds.cpu().numpy(), confidences.cpu().numpy()


class HandStateManager:
    """
    手势状态管理器
    负责跟踪手势状态变化，只有置信度超过阈值才认为手势有效
    """

    def __init__(self, confidence_threshold=0.77, stability_frames=3, idle_timeout=2.0):
        """
        Args:
            confidence_threshold: 置信度阈值，超过此值才认为手势有效（默认0.77）
            stability_frames: 稳定帧数，连续多少帧相同手势才确认状态变化（防止抖动）
            idle_timeout: 空闲超时时间（秒），超过此时间无有效手势自动进入空闲状态
        """
        self.confidence_threshold = confidence_threshold
        self.stability_frames = stability_frames
        self.idle_timeout = idle_timeout

        # 当前状态
        self.current_state = None  # 当前有效手势的类别ID
        self.current_confidence = 0.0  # 当前手势的置信度
        self.last_valid_time = time.time()  # 上次检测到有效手势的时间

        # 用于稳定性判断的缓冲区
        self.prediction_buffer = deque(maxlen=stability_frames)
        self.confidence_buffer = deque(maxlen=stability_frames)

        # 回调函数（可选）
        self.on_state_change = None  # 状态变化时的回调
        self.on_idle = None  # 进入空闲状态时的回调

        # 状态记录
        self.state_history = []  # 记录状态变化历史

    def update(self, class_id, confidence):
        """
        更新状态管理器
        Args:
            class_id: 模型预测的类别ID
            confidence: 模型预测的置信度
        Returns:
            current_hand_state: 当前有效手势（如果置信度足够且稳定）
            is_valid: 是否检测到有效手势
        """
        # 判断当前预测是否有效
        is_valid_prediction = confidence >= self.confidence_threshold

        if is_valid_prediction:
            # 有效预测，添加到缓冲区
            self.prediction_buffer.append(class_id)
            self.confidence_buffer.append(confidence)

            # 检查缓冲区是否已满
            if len(self.prediction_buffer) == self.stability_frames:
                # 检查是否所有预测都相同
                unique_predictions = set(self.prediction_buffer)
                if len(unique_predictions) == 1:
                    # 稳定的有效手势
                    stable_class_id = self.prediction_buffer[0]
                    avg_confidence = np.mean(self.confidence_buffer)

                    # 检查是否发生状态变化
                    if self.current_state != stable_class_id:
                        self._change_state(stable_class_id, avg_confidence)

                    self.last_valid_time = time.time()
                    return stable_class_id, True
        else:
            # 无效预测，清空缓冲区
            self.prediction_buffer.clear()
            self.confidence_buffer.clear()

        # 检查是否超时进入空闲状态
        if self.current_state is not None:
            if time.time() - self.last_valid_time > self.idle_timeout:
                self._enter_idle()

        return self.current_state, self.current_state is not None

    def _change_state(self, new_state, confidence):
        """状态变化时的处理"""
        old_state = self.current_state
        self.current_state = new_state
        self.current_confidence = confidence

        # 记录状态变化
        change_record = {
            'timestamp': time.time(),
            'from_state': old_state,
            'to_state': new_state,
            'confidence': confidence
        }
        self.state_history.append(change_record)

        # 打印状态变化信息
        if old_state is None:
            print(f"[手势识别] 检测到新手势: {new_state} (置信度: {confidence:.2%})")
        else:
            print(f"[手势识别] 手势变化: {old_state} -> {new_state} (置信度: {confidence:.2%})")

        # 触发回调
        if self.on_state_change:
            self.on_state_change(old_state, new_state, confidence)

    def _enter_idle(self):
        """进入空闲状态"""
        old_state = self.current_state
        self.current_state = None
        self.current_confidence = 0.0

        # 记录空闲状态
        change_record = {
            'timestamp': time.time(),
            'from_state': old_state,
            'to_state': None,
            'confidence': 0.0,
            'is_idle': True
        }
        self.state_history.append(change_record)

        print(f"[手势识别] 进入空闲状态 (最后手势: {old_state}, 超时: {self.idle_timeout}秒)")

        # 触发回调
        if self.on_idle:
            self.on_idle(old_state)

    def get_current_state(self):
        """获取当前有效手势"""
        return self.current_state

    def is_idle(self):
        """是否处于空闲状态"""
        return self.current_state is None

    def get_state_duration(self):
        """获取当前状态的持续时间（秒）"""
        if self.current_state is None:
            return 0.0
        return time.time() - self.last_valid_time

    def reset(self):
        """重置状态管理器"""
        self.current_state = None
        self.current_confidence = 0.0
        self.prediction_buffer.clear()
        self.confidence_buffer.clear()
        self.last_valid_time = time.time()
        print("[手势识别] 状态管理器已重置")


def main():
    # 初始化分类器
    model_path = "runs/train/20260422_083031/best_model.pth"  # 修改为你的模型路径
    classifier = HandClassifier(model_path, num_classes=5)

    # 初始化状态管理器
    state_manager = HandStateManager(
        confidence_threshold=0.77,  # 置信度阈值77%
        stability_frames=3,  # 连续3帧确认
        idle_timeout=2.0  # 2秒无手势进入空闲
    )

    # 可选：设置回调函数
    def on_hand_change(old_state, new_state, confidence):
        """手势变化时的回调"""
        # 这里可以添加自定义逻辑，如播放声音、更新UI等
        pass

    def on_idle(last_state):
        """进入空闲时的回调"""
        # 这里可以添加空闲时的处理逻辑
        pass

    state_manager.on_state_change = on_hand_change
    state_manager.on_idle = on_idle

    # 打开摄像头
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("按 'q' 退出")
    print("按 'r' 重置状态管理器")
    print("按 's' 保存当前帧")
    print("\n手势识别规则：")
    print(f"- 置信度 > {state_manager.confidence_threshold * 100:.0f}% 才认为手势有效")
    print(f"- 连续 {state_manager.stability_frames} 帧相同手势才确认")
    print(f"- {state_manager.idle_timeout} 秒无有效手势自动进入空闲状态\n")

    # FPS计算
    fps_counter = deque(maxlen=30)
    last_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法获取摄像头画面")
            break

        # 推理
        class_id, confidence, all_probs = classifier.predict(frame)

        # 更新状态管理器
        current_hand, is_valid = state_manager.update(class_id, confidence)

        # 计算FPS
        current_time = time.time()
        fps_counter.append(1.0 / (current_time - last_time + 1e-6))
        fps = np.mean(fps_counter)
        last_time = current_time

        # 在图像上绘制结果
        # 显示当前有效手势（超过77%置信度且稳定的）
        if is_valid and current_hand is not None:
            class_name = classifier.class_names[current_hand]
            status_text = f"ACTIVE: {class_name} (Conf: {state_manager.current_confidence:.2%})"
            status_color = (0, 255, 0)  # 绿色
        else:
            status_text = "IDLE (No valid gesture)"
            status_color = (0, 0, 255)  # 红色

        cv2.putText(frame, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        # 显示原始预测结果
        raw_text = f"Raw: {classifier.class_names[class_id]} ({confidence:.2%})"
        raw_color = (0, 255, 0) if confidence >= 0.77 else (255, 255, 255)
        cv2.putText(frame, raw_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, raw_color, 1)

        # 显示状态信息
        info_y = 90
        cv2.putText(frame, f"State: {current_hand if current_hand is not None else 'IDLE'}",
                    (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        duration = state_manager.get_state_duration()
        cv2.putText(frame, f"Duration: {duration:.1f}s",
                    (10, info_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        cv2.putText(frame, f"FPS: {fps:.1f}",
                    (10, info_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # 显示所有类别的概率（高亮超过阈值的）
        threshold_y = 140
        cv2.putText(frame, f"Confidence threshold: {state_manager.confidence_threshold * 100:.0f}%",
                    (10, threshold_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        for i in range(classifier.num_classes):
            prob = all_probs[i]
            is_above_threshold = prob >= state_manager.confidence_threshold
            color = (0, 255, 0) if is_above_threshold else (200, 200, 200)
            marker = ">" if is_above_threshold else " "
            prob_text = f"{marker} Class {i} ({classifier.class_names[i]}): {prob:.2%}"
            cv2.putText(frame, prob_text, (10, threshold_y + 25 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        cv2.imshow('Hand Gesture Recognition', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite("saved_frame.jpg", frame)
            print("已保存当前帧")
        elif key == ord('r'):
            state_manager.reset()
            print("状态管理器已重置")

    cap.release()
    cv2.destroyAllWindows()

    # 打印状态变化历史
    print("\n=== 状态变化历史 ===")
    for record in state_manager.state_history:
        if record.get('is_idle'):
            print(f"[空闲] 从手势 {record['from_state']} 进入空闲")
        else:
            print(f"[变化] {record['from_state']} -> {record['to_state']} "
                  f"(置信度: {record['confidence']:.2%})")


if __name__ == "__main__":
    main()