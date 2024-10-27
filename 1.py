import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import tempfile
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from pathlib import Path

class PitcherFormAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
    def calculate_angle(self, a, b, c):
        """세 점 사이의 각도 계산"""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(np.degrees(radians))
        
        if angle > 180:
            angle = 360 - angle
            
        return angle
    
    def calculate_velocity(self, point1, point2, fps):
        """두 점 사이의 속도 계산"""
        distance = np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
        velocity = distance * fps  # 픽셀/초 단위
        return velocity
    
    def analyze_frame(self, frame, frame_idx, fps):
        """단일 프레임 분석"""
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image)
        
        frame_data = {
            'frame_number': frame_idx,
            'timestamp': frame_idx / fps,
            'elbow_angle': None,
            'shoulder_angle': None,
            'hip_angle': None,
            'knee_angle': None,
            'arm_velocity': None
        }
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # 주요 관절 좌표 추출
            shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                       landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                    landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist = [landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                    landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            hip = [landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                  landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            knee = [landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                   landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            ankle = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                    landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            
            # 각도 계산
            frame_data['elbow_angle'] = self.calculate_angle(shoulder, elbow, wrist)
            frame_data['shoulder_angle'] = self.calculate_angle(hip, shoulder, elbow)
            frame_data['hip_angle'] = self.calculate_angle(shoulder, hip, knee)
            frame_data['knee_angle'] = self.calculate_angle(hip, knee, ankle)
            
            # 팔 속도 계산 (투구 방향으로의 속도)
            frame_data['arm_velocity'] = self.calculate_velocity(elbow, wrist, fps)
            
            # 포즈 마커 그리기
            self.mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )
            
            # 각도 표시
            cv2.putText(image, f'Elbow: {int(frame_data["elbow_angle"])}°', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(image, f'Shoulder: {int(frame_data["shoulder_angle"])}°', 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR), frame_data

def analyze_pitching_phases(df):
    """투구 단계 분석"""
    phases = {
        'wind_up': {'start': 0, 'end': 0},
        'stride': {'start': 0, 'end': 0},
        'arm_cocking': {'start': 0, 'end': 0},
        'acceleration': {'start': 0, 'end': 0},
        'release': {'start': 0, 'end': 0}
    }
    
    # 각 구간의 시작과 끝 프레임 찾기
    max_knee_angle_idx = df['knee_angle'].idxmax()
    min_elbow_angle_idx = df['elbow_angle'].idxmin()
    max_shoulder_angle_idx = df['shoulder_angle'].idxmax()
    max_arm_velocity_idx = df['arm_velocity'].idxmax()
    
    phases['wind_up']['end'] = max_knee_angle_idx
    phases['stride']['start'] = max_knee_angle_idx
    phases['stride']['end'] = min_elbow_angle_idx
    phases['arm_cocking']['start'] = min_elbow_angle_idx
    phases['arm_cocking']['end'] = max_shoulder_angle_idx
    phases['acceleration']['start'] = max_shoulder_angle_idx
    phases['acceleration']['end'] = max_arm_velocity_idx
    phases['release']['start'] = max_arm_velocity_idx
    phases['release']['end'] = len(df) - 1
    
    return phases

def create_analysis_plots(df, phases):
    """분석 그래프 생성"""
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('관절 각도', '팔 속도', '투구 단계'),
        vertical_spacing=0.12
    )
    
    # 관절 각도 그래프
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['elbow_angle'], name='팔꿈치 각도',
                  line=dict(color='red')), row=1, col=1)
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['shoulder_angle'], name='어깨 각도',
                  line=dict(color='blue')), row=1, col=1)
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['knee_angle'], name='무릎 각도',
                  line=dict(color='green')), row=1, col=1)
    
    # 속도 그래프
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['arm_velocity'], name='팔 속도',
                  line=dict(color='purple')), row=2, col=1)
    
    # 투구 단계 표시
    phase_colors = {'wind_up': 'rgba(255,0,0,0.3)', 
                   'stride': 'rgba(0,255,0,0.3)',
                   'arm_cocking': 'rgba(0,0,255,0.3)',
                   'acceleration': 'rgba(255,255,0,0.3)',
                   'release': 'rgba(128,0,128,0.3)'}
    
    for phase, timing in phases.items():
        start_time = df.loc[timing['start'], 'timestamp']
        end_time = df.loc[timing['end'], 'timestamp']
        
        fig.add_vrect(
            x0=start_time, x1=end_time,
            fillcolor=phase_colors[phase],
            opacity=0.5,
            layer="below",
            line_width=0,
            row=3, col=1
        )
        
        # 단계 이름 표시
        fig.add_annotation(
            x=(start_time + end_time)/2,
            y=0.5,
            text=phase.replace('_', ' ').title(),
            showarrow=False,
            row=3, col=1
        )
    
    fig.update_layout(height=800, title_text="투구 동작 분석")
    fig.update_xaxes(title_text="시간 (초)")
    fig.update_yaxes(title_text="각도 (도)", row=1, col=1)
    fig.update_yaxes(title_text="속도", row=2, col=1)
    
    return fig

def main():
    st.title("투수 폼 분석기")
    st.write("투구 동작이 담긴 동영상을 업로드하면 자세를 분석합니다.")
    
    uploaded_file = st.file_uploader("동영상 파일을 선택하세요", type=['mp4', 'avi', 'mov'])
    
    if uploaded_file is not None:
        # 임시 파일로 저장
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        
        analyzer = PitcherFormAnalyzer()
        video_file = tfile.name
        cap = cv2.VideoCapture(video_file)
        
        # 비디오 정보
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        st.write(f"프레임 수: {total_frames}, FPS: {fps}")
        
        # 진행 상태 바
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 분석 결과를 저장할 리스트
        frames_data = []
        
        # 비디오 플레이어
        video_placeholder = st.empty()
        
        # 프레임 단위 분석
        for frame_idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
                
            # 프레임 분석
            processed_frame, frame_data = analyzer.analyze_frame(frame, frame_idx, fps)
            frames_data.append(frame_data)
            
            # 진행 상태 업데이트
            progress = (frame_idx + 1) / total_frames
            progress_bar.progress(progress)
            status_text.text(f"분석 중... {int(progress * 100)}%")
            
            # 처리된 프레임 표시
            video_placeholder.image(processed_frame, channels="BGR", use_column_width=True)
            
        cap.release()
        
        # 분석 완료
        status_text.text("분석 완료!")
        
        # 데이터프레임 생성
        df = pd.DataFrame(frames_data)
        
        # 투구 단계 분석
        phases = analyze_pitching_phases(df)
        
        # 분석 결과 표시
        st.subheader("투구 동작 분석 결과")
        
        # 그래프 표시
        st.plotly_chart(create_analysis_plots(df, phases), use_container_width=True)
        
        # 주요 지표 분석
        st.subheader("주요 지표")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("최대 팔꿈치 각도", f"{df['elbow_angle'].max():.1f}°")
            st.metric("최대 어깨 각도", f"{df['shoulder_angle'].max():.1f}°")
            
        with col2:
            st.metric("최대 무릎 각도", f"{df['knee_angle'].max():.1f}°")
            st.metric("최대 팔 속도", f"{df['arm_velocity'].max():.1f}")
            
        with col3:
            st.metric("투구 소요 시간", f"{df['timestamp'].max():.2f}초")
        
        # CSV 다운로드 버튼
        csv = df.to_csv(index=False)
        st.download_button(
            label="분석 데이터 다운로드 (CSV)",
            data=csv,
            file_name='pitching_analysis.csv',
            mime='text/csv',
        )
        
        # 임시 파일 삭제
        Path(video_file).unlink()

if __name__ == "__main__":
    main()