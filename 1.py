import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import tempfile
import os
from PIL import Image
import matplotlib.pyplot as plt

class PitcherFormAnalyzer:
    def __init__(self):
        # 임시 디렉토리를 환경 변수로 설정
        self.temp_dir = tempfile.gettempdir()
        os.environ['MEDIAPIPE_MODEL_PATH'] = self.temp_dir
        
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 관절 각도 임계값 설정
        self.angle_thresholds = {
            'shoulder': {'min': 80, 'max': 100},
            'elbow': {'min': 85, 'max': 95},
            'hip': {'min': 170, 'max': 180},
            'knee': {'min': 170, 'max': 180}
        }

    def calculate_angle(self, a, b, c):
        """세 점 사이의 각도를 계산합니다."""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle > 180.0:
            angle = 360-angle
            
        return angle

    def analyze_frame(self, frame):
        """프레임에서 자세를 분석하고 피드백을 생성합니다."""
        # BGR을 RGB로 변환
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image)
        
        feedback = []
        angles = {}
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # 어깨 각도 계산
            shoulder_right = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                            landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow_right = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                          landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist_right = [landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                          landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            
            shoulder_angle = self.calculate_angle(shoulder_right, elbow_right, wrist_right)
            angles['shoulder'] = shoulder_angle
            
            # 팔꿈치 각도
            elbow_angle = self.calculate_angle(shoulder_right, elbow_right, wrist_right)
            angles['elbow'] = elbow_angle
            
            # 힙 각도
            hip_right = [landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                        landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            knee_right = [landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                         landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            ankle_right = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                          landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            
            hip_angle = self.calculate_angle(hip_right, knee_right, ankle_right)
            angles['hip'] = hip_angle
            
            # 무릎 각도
            knee_angle = self.calculate_angle(hip_right, knee_right, ankle_right)
            angles['knee'] = knee_angle
            
            # 피드백 생성
            if shoulder_angle < self.angle_thresholds['shoulder']['min']:
                feedback.append(f"어깨 각도가 너무 작습니다 ({shoulder_angle:.1f}°)")
            elif shoulder_angle > self.angle_thresholds['shoulder']['max']:
                feedback.append(f"어깨 각도가 너무 큽니다 ({shoulder_angle:.1f}°)")
                
            if elbow_angle < self.angle_thresholds['elbow']['min']:
                feedback.append(f"팔꿈치 각도가 너무 작습니다 ({elbow_angle:.1f}°)")
            elif elbow_angle > self.angle_thresholds['elbow']['max']:
                feedback.append(f"팔꿈치 각도가 너무 큽니다 ({elbow_angle:.1f}°)")
                
            if hip_angle < self.angle_thresholds['hip']['min']:
                feedback.append(f"힙 각도가 너무 작습니다 ({hip_angle:.1f}°)")
            elif hip_angle > self.angle_thresholds['hip']['max']:
                feedback.append(f"힙 각도가 너무 큽니다 ({hip_angle:.1f}°)")
                
            if knee_angle < self.angle_thresholds['knee']['min']:
                feedback.append(f"무릎 각도가 너무 작습니다 ({knee_angle:.1f}°)")
            elif knee_angle > self.angle_thresholds['knee']['max']:
                feedback.append(f"무릎 각도가 너무 큽니다 ({knee_angle:.1f}°)")
            
            # 랜드마크 그리기
            self.mp_draw.draw_landmarks(
                image, 
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_draw.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                self.mp_draw.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )
            
        return image, feedback, angles

def main():
    st.title("⚾ 야구 투수 자세 분석기")
    st.write("영상을 업로드하여 투구 자세를 분석해보세요.")
    
    # 파일 업로더
    uploaded_file = st.file_uploader("동영상 파일을 선택하세요", type=['mp4', 'avi', 'mov'])
    
    if uploaded_file is not None:
        # 임시 파일로 저장
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        
        # 비디오 캡처 객체 생성
        cap = cv2.VideoCapture(tfile.name)
        analyzer = PitcherFormAnalyzer()
        
        # 프레임 수 계산
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current_frame = 0
        
        # 진행 상황 표시
        progress_bar = st.progress(0)
        frame_text = st.empty()
        
        # 분석 결과를 표시할 컨테이너
        result_container = st.container()
        
        all_feedback = []
        all_angles = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # 프레임 분석
            image, feedback, angles = analyzer.analyze_frame(frame)
            
            # 결과 저장
            if feedback:
                all_feedback.extend(feedback)
            if angles:
                all_angles.append(angles)
            
            # 진행 상황 업데이트
            current_frame += 1
            progress = int((current_frame / total_frames) * 100)
            progress_bar.progress(progress)
            frame_text.text(f"분석 진행중... {progress}%")
            
        cap.release()
        
        # 임시 파일 삭제
        os.unlink(tfile.name)
        
        # 분석 결과 표시
        with result_container:
            st.success("분석이 완료되었습니다!")
            
            if all_angles:
                st.subheader("📊 각도 분석 결과")
                # 각도 데이터를 DataFrame으로 변환
                angle_data = pd.DataFrame(all_angles)
                
                # 각 관절별 평균 각도
                mean_angles = angle_data.mean()
                for joint, angle in mean_angles.items():
                    st.write(f"{joint.capitalize()} 평균 각도: {angle:.1f}°")
                
                # 각도 변화 그래프
                st.line_chart(angle_data)
            
            if all_feedback:
                st.subheader("💡 자세 피드백")
                unique_feedback = list(set(all_feedback))  # 중복 제거
                for fb in unique_feedback:
                    st.write(f"- {fb}")

if __name__ == '__main__':
    main()