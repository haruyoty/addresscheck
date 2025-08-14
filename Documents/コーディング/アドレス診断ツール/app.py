import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import mediapipe as mp
import math
from PIL import Image
import io
import base64
from streamlit_image_coordinates import streamlit_image_coordinates
import json
from datetime import datetime


# ページ設定
st.set_page_config(
    page_title="ゴルフアドレス診断ツール",
    page_icon="⛳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 定数定義
BALL_TARGETS = {
    "Driver": 0.88,
    "3W": 0.82, "4W": 0.80, "5W": 0.78, "7W": 0.76,
    "3UT": 0.75, "4UT": 0.73, "5UT": 0.70, "6UT": 0.68,
    "3I": 0.65, "4I": 0.62, "5I": 0.58, "6I": 0.55, "7I": 0.52, "8I": 0.50, "9I": 0.48,
    "PW": 0.45, "AW": 0.43, "SW": 0.40, "LW": 0.38
}

BALL_TOLERANCE = 0.06

STANCE_TARGETS = {
    "IRON": [0.95, 1.05],
    "FWUT": [1.05, 1.20],
    "DR": [1.20, 1.40]
}

HAND_POS_OFFSETS = {
    "DR": [0.05, 0.08],
    "OTHER": [-0.05, 0.05]
}

HEAD_TARGETS = {
    "Driver": 0.52,  # ドライバーは両足のセンター（0.5）より少し右寄り（2%右）
    "OTHER": 0.50
}

HEAD_TOLERANCE = 0.05

SPINE_ANGLE_TARGET = [170, 180]

FORWARD_TILT_TARGETS = {
    "Driver": 30,
    "3W": 32, "4W": 33, "5W": 34, "7W": 35,
    "3UT": 35, "4UT": 36, "5UT": 37, "6UT": 38,
    "3I": 38, "4I": 39, "5I": 40, "6I": 41, "7I": 41, "8I": 42, "9I": 43,
    "PW": 45, "AW": 45, "SW": 45, "LW": 45
}

TILT_TOLERANCE = 5
PARALLEL_TOLERANCE = 3

# MediaPipeの初期化
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def get_club_category(club):
    """クラブカテゴリを取得"""
    if club == "Driver":
        return "DR"
    elif club in ["FW", "UT"]:
        return "FWUT"
    else:
        return "IRON"


def detect_ball_automatically(image, landmarks):
    """ボールを自動検出（複数手法で精度向上）"""
    try:
        # RGB画像とグレースケール変換
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 足首の位置を取得（検索範囲を限定）
        left_ankle = landmarks[27]
        right_ankle = landmarks[28]
        
        h, w = image.shape[:2]
        
        # 足首間の範囲でボール検索（より広範囲で検索）
        ankle_y = int((left_ankle[1] + right_ankle[1]) / 2 * h)
        left_x = int(min(left_ankle[0], right_ankle[0]) * w)
        right_x = int(max(left_ankle[0], right_ankle[0]) * w)
        foot_width = right_x - left_x
        
        # 検索範囲を動的に調整（足幅に基づく）
        search_margin_x = max(200, int(foot_width * 1.5))
        search_margin_y_up = 80
        search_margin_y_down = 200
        
        search_top = max(0, ankle_y - search_margin_y_up)
        search_bottom = min(h, ankle_y + search_margin_y_down)
        search_left = max(0, left_x - search_margin_x)
        search_right = min(w, right_x + search_margin_x)
        
        # 検索領域を切り出し
        search_region = gray[search_top:search_bottom, search_left:search_right]
        rgb_region = rgb_image[search_top:search_bottom, search_left:search_right]
        
        best_ball = None
        max_confidence = 0
        detection_info = {"detected": False, "method": "estimated", "confidence": 0}
        
        # 手法1: HoughCircles（円検出）- より精密なパラメータ調整
        try:
            # 画像前処理の強化
            blurred = cv2.GaussianBlur(search_region, (5, 5), 0)
            
            # エッジ強調
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(blurred, -1, kernel)
            
            # 複数のパラメータで円検出を試行（より幅広い範囲）
            param_sets = [
                {"param1": 100, "param2": 20, "minR": 3, "maxR": 60},
                {"param1": 80, "param2": 25, "minR": 2, "maxR": 45},
                {"param1": 120, "param2": 15, "minR": 5, "maxR": 70},
                {"param1": 90, "param2": 30, "minR": 4, "maxR": 35},
                {"param1": 60, "param2": 35, "minR": 2, "maxR": 25}
            ]
            
            for i, params in enumerate(param_sets):
                # パラメータセットに応じて画像を選択
                detection_image = sharpened if i < 3 else blurred
                
                circles = cv2.HoughCircles(
                    detection_image,
                    cv2.HOUGH_GRADIENT,
                    dp=1,
                    minDist=max(10, params["minR"]),
                    param1=params["param1"],
                    param2=params["param2"],
                    minRadius=params["minR"],
                    maxRadius=params["maxR"]
                )
                
                if circles is not None:
                    circles = np.round(circles[0, :]).astype("int")
                    
                    for (x, y, r) in circles:
                        # 明るさ（白さ）をチェック
                        mask = np.zeros_like(search_region)
                        cv2.circle(mask, (x, y), r, 255, -1)
                        brightness = cv2.mean(search_region, mask=mask)[0]
                        
                        # 白色度をチェック（RGB）
                        rgb_mask = np.zeros(rgb_region.shape[:2], dtype=np.uint8)
                        cv2.circle(rgb_mask, (x, y), r, 255, -1)
                        mean_color = cv2.mean(rgb_region, mask=rgb_mask)[:3]
                        whiteness = min(mean_color) / max(mean_color) if max(mean_color) > 0 else 0
                        
                        # 信頼度計算
                        confidence = int((brightness * 0.6 + whiteness * 100 * 0.4))
                        
                        if confidence > max_confidence and brightness > 120:
                            max_confidence = confidence
                            actual_x = (search_left + x) / w
                            actual_y = (search_top + y) / h
                            best_ball = [actual_x, actual_y]
                            detection_info = {
                                "detected": True,
                                "method": "hough_circles",
                                "confidence": min(100, confidence),
                                "radius": r,
                                "brightness": int(brightness)
                            }
        except Exception as e:
            print(f"円検出エラー: {e}")
        
        # 手法2: 白色物体検出（フォールバック）
        if max_confidence < 60:
            try:
                # HSV変換で白色検出
                hsv = cv2.cvtColor(rgb_region, cv2.COLOR_RGB2HSV)
                
                # 白色の範囲
                lower_white = np.array([0, 0, 180])
                upper_white = np.array([180, 30, 255])
                
                # 白色マスク
                white_mask = cv2.inRange(hsv, lower_white, upper_white)
                
                # 輪郭検出
                contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if 50 < area < 2000:  # 適切なサイズの物体
                        # 外接円を計算
                        (x, y), radius = cv2.minEnclosingCircle(contour)
                        
                        # 円形度チェック
                        perimeter = cv2.arcLength(contour, True)
                        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
                        
                        if circularity > 0.6:  # 円形に近い
                            confidence = int(circularity * 80 + (area / 1000) * 20)
                            
                            if confidence > max_confidence:
                                max_confidence = confidence
                                actual_x = (search_left + x) / w
                                actual_y = (search_top + y) / h
                                best_ball = [actual_x, actual_y]
                                detection_info = {
                                    "detected": True,
                                    "method": "white_detection",
                                    "confidence": min(100, confidence),
                                    "area": int(area)
                                }
            except Exception as e:
                print(f"白色検出エラー: {e}")
        
        if best_ball and max_confidence > 40:
            return best_ball, detection_info
    
    except Exception as e:
        print(f"ボール自動検出エラー: {e}")
    
    # デフォルト位置（足の中央前方）
    center_x = (landmarks[27][0] + landmarks[28][0]) / 2
    center_y = min(landmarks[27][1], landmarks[28][1]) + 0.15  # 足首より少し下
    
    return [center_x, center_y], {"detected": False, "method": "estimated", "confidence": 25}


def extract_landmarks(image):
    """MediaPipeを使用してランドマークを抽出"""
    try:
        with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5
        ) as pose:
            # RGB変換
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_image)
            
            if results.pose_landmarks:
                landmarks = []
                for lm in results.pose_landmarks.landmark:
                    landmarks.append([lm.x, lm.y, lm.z, lm.visibility])
                return np.array(landmarks), results
            else:
                return None, None
    except Exception as e:
        st.error(f"ランドマーク抽出エラー: {e}")
        return None, None


def calculate_distance(p1, p2):
    """2点間の距離を計算"""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def calculate_angle(p1, p2, p3):
    """3点で構成される角度を計算（度数）"""
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle_rad = math.acos(cos_angle)
    return math.degrees(angle_rad)


def evaluate_ball_position(landmarks, ball_pos, club, is_left_handed):
    """ボール位置の評価"""
    try:
        # 左右足首の座標取得（MediaPipeランドマーク）
        left_ankle = landmarks[27][:2]  # LEFT_ANKLE
        right_ankle = landmarks[28][:2]  # RIGHT_ANKLE
        
        if is_left_handed:
            left_ankle, right_ankle = right_ankle, left_ankle
            
        # 足幅線での相対位置計算
        foot_line_width = abs(left_ankle[0] - right_ankle[0])
        if foot_line_width == 0:
            return None, None, None, 0, "足幅を検出できません"
            
        ball_relative = (ball_pos[0] - right_ankle[0]) / foot_line_width
        
        target = BALL_TARGETS[club]
        error = abs(ball_relative - target)
        
        # スコア計算（より寛容な評価）
        if error <= BALL_TOLERANCE:
            score = 100
        elif error <= 0.15:
            score = 100 - (error - BALL_TOLERANCE) * 100  # より緩やかな減点
        else:
            score = max(60, 90 - (error - 0.15) * 100)  # 最低60点保証
            
        # コメント生成
        if score >= 90:
            comment = f"ボール位置は最適です（目標: {target:.2f}, 実測: {ball_relative:.2f}）"
        elif score >= 70:
            comment = f"ボール位置はやや調整が必要です（誤差: {error:.2f}）"
        else:
            if ball_relative < target:
                comment = f"ボール位置を左足側に約{abs(ball_relative - target)*100:.0f}%移動してください"
            else:
                comment = f"ボール位置を右足側に約{abs(ball_relative - target)*100:.0f}%移動してください"
                
        return ball_relative, target, error, score, comment
        
    except Exception as e:
        return None, None, None, 15, f"ボール位置評価エラー（最低点数付与）: {e}"


def evaluate_stance_width(landmarks, club, is_left_handed):
    """スタンス幅の評価"""
    try:
        # 足首と肩の座標取得
        left_ankle = landmarks[27][:2]
        right_ankle = landmarks[28][:2]
        left_shoulder = landmarks[11][:2]
        right_shoulder = landmarks[12][:2]
        
        if is_left_handed:
            left_ankle, right_ankle = right_ankle, left_ankle
            left_shoulder, right_shoulder = right_shoulder, left_shoulder
            
        stance_width = calculate_distance(left_ankle, right_ankle)
        shoulder_width = calculate_distance(left_shoulder, right_shoulder)
        
        if shoulder_width == 0:
            return None, None, None, 0, "肩幅を検出できません"
            
        stance_ratio = stance_width / shoulder_width
        
        # クラブカテゴリ別の目標範囲
        category = get_club_category(club)
        target_range = STANCE_TARGETS[category]
        target_center = (target_range[0] + target_range[1]) / 2
        
        # エラー計算
        if target_range[0] <= stance_ratio <= target_range[1]:
            error = 0
        else:
            error = min(abs(stance_ratio - target_range[0]), abs(stance_ratio - target_range[1]))
            
        # スコア計算（より寛容な評価）
        if error == 0:
            score = 100
        elif error <= 0.20:
            score = 100 - error * 50  # より緩やかな減点
        else:
            score = max(65, 90 - (error - 0.20) * 50)  # 最低65点保証
            
        # コメント生成
        if score >= 90:
            comment = f"スタンス幅は最適です（比率: {stance_ratio:.2f}）"
        elif stance_ratio < target_range[0]:
            comment = f"スタンスをもう少し広くしてください（現在: {stance_ratio:.2f}, 目標: {target_range[0]:.2f}-{target_range[1]:.2f}）"
        else:
            comment = f"スタンスをもう少し狭くしてください（現在: {stance_ratio:.2f}, 目標: {target_range[0]:.2f}-{target_range[1]:.2f}）"
            
        return stance_ratio, target_center, error, score, comment
        
    except Exception as e:
        return None, None, None, 15, f"スタンス幅評価エラー（最低点数付与）: {e}"


def evaluate_hand_position(landmarks, club, shooting_direction, is_left_handed):
    """手の位置の評価"""
    try:
        # 手の甲の座標取得（手首より指先方向を推定）
        left_wrist = landmarks[15][:2]
        right_wrist = landmarks[16][:2]
        left_index = landmarks[19][:2]  # 左手人差し指
        right_index = landmarks[20][:2]  # 右手人差し指
        
        # 手の甲位置を推定（手首と人差し指の中点）
        left_hand_back = [(left_wrist[0] + left_index[0]) / 2, (left_wrist[1] + left_index[1]) / 2]
        right_hand_back = [(right_wrist[0] + right_index[0]) / 2, (right_wrist[1] + right_index[1]) / 2]
        
        if shooting_direction == "後方":
            # 後方撮影時：手と体の距離を拳の個数で評価
            left_shoulder = landmarks[11][:2]
            right_shoulder = landmarks[12][:2]
            left_hip = landmarks[23][:2]
            right_hip = landmarks[24][:2]
            
            if is_left_handed:
                left_hand_back, right_hand_back = right_hand_back, left_hand_back
                left_shoulder, right_shoulder = right_shoulder, left_shoulder
                left_hip, right_hip = right_hip, left_hip
            
            # 体の中心線（肩と腰の中心）
            shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) / 2
            hip_center_x = (left_hip[0] + right_hip[0]) / 2
            body_center_x = (shoulder_center_x + hip_center_x) / 2
            
            # 手の中点位置
            actual_hand_x = (left_hand_back[0] + right_hand_back[0]) / 2
            
            # 手と体の距離
            hand_body_distance = abs(actual_hand_x - body_center_x)
            
            # 肩幅で正規化（拳のサイズの基準）
            shoulder_width = abs(left_shoulder[0] - right_shoulder[0])
            if shoulder_width == 0:
                return None, None, None, 0, "肩幅を検出できません"
                
            # 拳1個分の推定サイズ（肩幅の約8%）
            fist_size = shoulder_width * 0.08
            
            # 距離を拳の個数で換算
            distance_in_fists = hand_body_distance / fist_size
            
            # クラブ別の理想距離（拳の個数）
            if club == "Driver":
                # ドライバー：2～3個分
                target_min = 2.0
                target_max = 3.0
                target_center = 2.5
                target_description = "拳2～3個分"
            else:
                # その他：1.5～2個分
                target_min = 1.5
                target_max = 2.0
                target_center = 1.75
                target_description = "拳1.5～2個分"
            
            # エラー計算
            if target_min <= distance_in_fists <= target_max:
                error = 0
            else:
                error = min(abs(distance_in_fists - target_min), abs(distance_in_fists - target_max))
            
            # スコア計算
            if error == 0:
                score = 100
            elif error <= 0.3:  # 0.3個分以内の誤差
                score = 100 - error * 50
            elif error <= 0.7:  # 0.7個分以内の誤差
                score = 85 - (error - 0.3) * 50
            else:
                score = max(65, 65 - (error - 0.7) * 30)
                
            # コメント生成
            if score >= 90:
                comment = f"手と体の距離は理想的です（{target_description}）"
            else:
                if distance_in_fists < target_min:
                    comment = f"手を体から少し離してください（現在: 拳{distance_in_fists:.1f}個分, 理想: {target_description}）"
                else:
                    comment = f"手を体に少し近づけてください（現在: 拳{distance_in_fists:.1f}個分, 理想: {target_description}）"
                    
            return error, 0.3, error, score, comment
        
        # 太ももの位置取得
        left_hip = landmarks[23][:2]
        right_hip = landmarks[24][:2]
        left_knee = landmarks[25][:2]
        right_knee = landmarks[26][:2]
        
        if is_left_handed:
            left_hand_back, right_hand_back = right_hand_back, left_hand_back
            left_hip, right_hip = right_hip, left_hip
            left_knee, right_knee = right_knee, left_knee
            
        # 腰の位置のセンターを計算
        hip_center_x = (left_hip[0] + right_hip[0]) / 2
        hip_center_y = (left_hip[1] + right_hip[1]) / 2
        
        # 太ももの幅を推定（腰と膝の距離から）
        hip_width = abs(left_hip[0] - right_hip[0])
        
        # 理想的な手の位置：腰のセンターより少し左で、左足太ももの内側
        # 腰のセンターから左に少しずらした位置を理想とする
        left_offset = hip_width * 0.08  # 腰のセンターから8%左寄り
        
        # Y座標は左足太ももの高さ（左腰と左膝の中間）
        left_thigh_y = (left_hip[1] + left_knee[1]) / 2
        
        # 理想的な手の位置：腰のセンターより少し左、左足太ももの内側の高さ
        target_hand_x = hip_center_x - left_offset  # 腰のセンターから少し左
        target_hand_y = left_thigh_y  # 左足太ももの高さ
        
        # 実際の手の中点位置
        actual_hand_x = (left_hand_back[0] + right_hand_back[0]) / 2
        actual_hand_y = (left_hand_back[1] + right_hand_back[1]) / 2
        
        # 左右の位置のみを評価（高さは無視）
        distance_x = abs(actual_hand_x - target_hand_x)
        
        # 正規化（腰幅で割る）
        normalized_error_x = distance_x / hip_width if hip_width > 0 else distance_x
        
        # 誤差は左右の位置のみ
        total_error = normalized_error_x
        
        # スコア計算（より寛容な評価）
        if total_error <= 0.05:
            score = 100
        elif total_error <= 0.15:
            score = 100 - (total_error - 0.05) * 200  # より緩やかな減点
        elif total_error <= 0.30:
            score = 80 - (total_error - 0.15) * 100   # 中程度の誤差でも80-65点
        else:
            score = max(55, 65 - (total_error - 0.30) * 50)  # 最低55点保証
            
        # コメント生成（左右の位置のみ）
        if score >= 90:
            comment = "手の位置は理想的です（左足太もも内側）"
        else:
            # 左右の方向を判定
            if actual_hand_x < target_hand_x:
                direction_x = "右"
            else:
                direction_x = "左"
                
            comment = f"手をもう少し{direction_x}に移動してください（左足太もも内側が理想位置）"
                
        return total_error, 0.1, total_error, score, comment
        
    except Exception as e:
        return None, None, None, 20, f"手の位置評価エラー（最低点数付与）: {e}"


def evaluate_weight_distribution(landmarks, club, is_left_handed, shooting_direction="正面"):
    """重心配分・重心の位置の評価"""
    try:
        if shooting_direction == "後方":
            # 後方撮影時：重心位置の総合判定（複数要素を組み合わせ）
            left_shoulder = landmarks[11][:2]
            right_shoulder = landmarks[12][:2]
            left_ankle = landmarks[27][:2]
            right_ankle = landmarks[28][:2]
            left_hip = landmarks[23][:2]
            right_hip = landmarks[24][:2]
            left_knee = landmarks[25][:2]
            right_knee = landmarks[26][:2]
            
            if is_left_handed:
                left_shoulder, right_shoulder = right_shoulder, left_shoulder
                left_ankle, right_ankle = right_ankle, left_ankle
                left_hip, right_hip = right_hip, left_hip
                left_knee, right_knee = right_knee, left_knee
            
            # 1. 肩の中心位置
            shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) / 2
            
            # 2. 腰の中心位置
            hip_center_x = (left_hip[0] + right_hip[0]) / 2
            
            # 3. 膝の中心位置
            knee_center_x = (left_knee[0] + right_knee[0]) / 2
            
            # 4. 母指球の推定位置
            foot_length_estimate = 0.08
            ball_of_foot_left_x = left_ankle[0] - foot_length_estimate
            ball_of_foot_right_x = right_ankle[0] - foot_length_estimate
            ball_of_foot_center_x = (ball_of_foot_left_x + ball_of_foot_right_x) / 2
            
            # 重心線の推定（肩→腰→膝の重心線）
            body_center_line_x = (shoulder_center_x + hip_center_x + knee_center_x) / 3
            
            # 母指球との位置関係
            shoulder_width = abs(left_shoulder[0] - right_shoulder[0])
            if shoulder_width == 0:
                return None, None, None, 0, "肩幅を検出できません"
                
            # 体の重心線と母指球の位置差
            deviation = abs(body_center_line_x - ball_of_foot_center_x) / shoulder_width
            
            # スコア計算（より甘い評価）
            if deviation <= 0.08:  # 非常に良い（範囲拡大）
                score = 100
            elif deviation <= 0.20:  # 良い（範囲拡大）
                score = 100 - deviation * 150  # より緩やかな減点
            elif deviation <= 0.35:  # 普通（新しい段階）
                score = 100 - (0.20 * 150) - (deviation - 0.20) * 80  # さらに緩やか
            else:
                score = max(75, 85 - (deviation - 0.35) * 50)  # 最低75点保証
            
            # コメント生成
            if score >= 90:
                comment = "重心の位置は理想的です（体の中心軸が母指球の真上）"
            else:
                if body_center_line_x < ball_of_foot_center_x:
                    # 体の重心線が母指球よりもつま先寄り（手前）= つま先重心すぎ
                    comment = "つま先重心になっています。重心をかかと側に移動してください（体の重心線が母指球の真上が理想）"
                else:
                    # 体の重心線が母指球よりもかかと寄り（奥）= かかと重心
                    comment = "かかと重心になっています。重心をつま先側に移動してください（体の重心線が母指球の真上が理想）"
            
            return deviation, 0.1, deviation, score, comment
            
        else:
            # 正面撮影時：従来の重心配分評価
            left_hip = landmarks[23][:2]
            right_hip = landmarks[24][:2]
            
            if is_left_handed:
                left_hip, right_hip = right_hip, left_hip
                
            # 重心位置を股関節の中心からの偏差で推定（簡易計算）
            hip_center_x = (left_hip[0] + right_hip[0]) / 2
            hip_width = abs(left_hip[0] - right_hip[0])
            
            if hip_width == 0:
                return None, None, None, 0, "股関節の幅を検出できません"
            
            # 体の傾きから重心配分を推定
            shoulders_center = (landmarks[11][0] + landmarks[12][0]) / 2
            weight_shift = (shoulders_center - hip_center_x) / hip_width
            
            # 理想的な重心配分を定義
            if club == "Driver":
                target_shift = 0.1  # 右にわずかに偏る
                ideal_comment = "1Wの場合右足重心（6:4）"
            else:
                target_shift = 0.0  # 中央
                ideal_comment = "左右均等（5:5）"
            
            error = abs(weight_shift - target_shift)
            
            # スコア計算
            if error <= 0.05:
                score = 100
            elif error <= 0.1:
                score = 100 - error * 200
            else:
                score = max(70, 90 - (error - 0.1) * 150)
            
            # コメント生成
            if score >= 90:
                comment = f"重心配分は理想的です（{ideal_comment}）"
            else:
                if weight_shift < target_shift:
                    if club == "Driver":
                        comment = f"もう少し右足に体重をかけてください（理想：{ideal_comment}）"
                    else:
                        comment = f"もう少し右足に体重をかけてください（理想：{ideal_comment}）"
                else:
                    if club == "Driver":
                        comment = f"体重が右足にかかりすぎています（理想：{ideal_comment}）"
                    else:
                        comment = f"もう少し左足に体重をかけてください（理想：{ideal_comment}）"
            
            return error, 0.1, error, score, comment
        
    except Exception as e:
        return None, None, None, 30, f"重心配分評価エラー（最低点数付与）: {e}"


def evaluate_head_position(landmarks, club, is_left_handed):
    """頭の位置の評価"""
    try:
        # 頭部と足首の座標取得
        nose = landmarks[0][:2]
        left_ankle = landmarks[27][:2]
        right_ankle = landmarks[28][:2]
        
        if is_left_handed:
            left_ankle, right_ankle = right_ankle, left_ankle
            
        foot_width = abs(left_ankle[0] - right_ankle[0])
        if foot_width == 0:
            return None, None, None, 0, "足幅を検出できません"
            
        head_relative = (nose[0] - right_ankle[0]) / foot_width
        
        if club == "Driver":
            target = HEAD_TARGETS["Driver"]
        else:
            target = HEAD_TARGETS["OTHER"]
            
        error = abs(head_relative - target)
        
        # スコア計算（より寛容な評価）
        if error <= HEAD_TOLERANCE:
            score = 100
        elif error <= 0.15:
            score = 100 - (error - HEAD_TOLERANCE) * 150  # より緩やかな減点
        else:
            score = max(65, 85 - (error - 0.15) * 100)  # 最低65点保証
            
        # コメント生成
        if score >= 90:
            if club == "Driver":
                comment = f"頭の位置は最適です（両足のセンターより少し右）"
            else:
                comment = f"頭の位置は最適です（両足のセンター）"
        elif head_relative < target:
            if club == "Driver":
                comment = f"頭をもう少し右に寄せてください（両足のセンターより少し右が理想）"
            else:
                comment = f"頭をもう少し右に寄せてください（両足のセンターが理想）"
        else:
            comment = f"頭をもう少し左に寄せてください"
            
        return head_relative, target, error, score, comment
        
    except Exception as e:
        return None, None, None, 15, f"頭の位置評価エラー（最低点数付与）: {e}"


def evaluate_body_alignment(landmarks, is_left_handed):
    """体の向き（肩、腰、かかと）の評価"""
    try:
        # かかと（足首）、腰、肩の座標取得
        left_ankle = landmarks[27][:2]  # かかと方向
        right_ankle = landmarks[28][:2]
        left_hip = landmarks[23][:2]
        right_hip = landmarks[24][:2]
        left_shoulder = landmarks[11][:2]
        right_shoulder = landmarks[12][:2]
        
        if is_left_handed:
            left_ankle, right_ankle = right_ankle, left_ankle
            left_hip, right_hip = right_hip, left_hip
            left_shoulder, right_shoulder = right_shoulder, left_shoulder
            
        # 各ラインの角度計算
        heel_angle = math.degrees(math.atan2(left_ankle[1] - right_ankle[1], left_ankle[0] - right_ankle[0]))
        hip_angle = math.degrees(math.atan2(left_hip[1] - right_hip[1], left_hip[0] - right_hip[0]))
        shoulder_angle = math.degrees(math.atan2(left_shoulder[1] - right_shoulder[1], left_shoulder[0] - right_shoulder[0]))
        
        # 角度差の計算
        heel_hip_diff = abs(heel_angle - hip_angle)
        hip_shoulder_diff = abs(hip_angle - shoulder_angle)
        heel_shoulder_diff = abs(heel_angle - shoulder_angle)
        
        # 180度を超える場合の補正
        heel_hip_diff = min(heel_hip_diff, 180 - heel_hip_diff)
        hip_shoulder_diff = min(hip_shoulder_diff, 180 - hip_shoulder_diff)
        heel_shoulder_diff = min(heel_shoulder_diff, 180 - heel_shoulder_diff)
        
        max_diff = max(heel_hip_diff, hip_shoulder_diff, heel_shoulder_diff)
        
        # スコア計算（より寛容な評価）
        if max_diff <= PARALLEL_TOLERANCE:
            score = 100
        elif max_diff <= 10:
            score = 100 - (max_diff - PARALLEL_TOLERANCE) * 3  # より緩やかな減点
        else:
            score = max(65, 80 - (max_diff - 10) * 2)  # 最低65点保証
            
        # コメント生成
        if score >= 90:
            comment = f"肩、腰、かかとの向きは適切に揃っています"
        else:
            # 最も大きなずれを特定してアドバイス
            if max_diff == heel_hip_diff:
                comment = f"かかとと腰の向きを揃えてください（誤差: {heel_hip_diff:.1f}°）"
            elif max_diff == hip_shoulder_diff:
                comment = f"腰と肩の向きを揃えてください（誤差: {hip_shoulder_diff:.1f}°）"
            else:
                comment = f"かかとと肩の向きを揃えてください（誤差: {heel_shoulder_diff:.1f}°）"
            
        return max_diff, PARALLEL_TOLERANCE, max_diff, score, comment
        
    except Exception as e:
        return None, None, None, 15, f"体の向き評価エラー（最低点数付与）: {e}"


def evaluate_spine_posture(landmarks):
    """背中の曲がり（猫背）の評価"""
    try:
        # 耳、肩、腰の座標取得
        left_ear = landmarks[7][:2]
        right_ear = landmarks[8][:2] 
        left_shoulder = landmarks[11][:2]
        right_shoulder = landmarks[12][:2]
        left_hip = landmarks[23][:2]
        right_hip = landmarks[24][:2]
        
        # 各部位の中点
        ear_center = [(left_ear[0] + right_ear[0]) / 2, (left_ear[1] + right_ear[1]) / 2]
        shoulder_center = [(left_shoulder[0] + right_shoulder[0]) / 2, (left_shoulder[1] + right_shoulder[1]) / 2]
        hip_center = [(left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2]
        
        # 肩での角度計算
        spine_angle = calculate_angle(ear_center, shoulder_center, hip_center)
        
        target_min, target_max = SPINE_ANGLE_TARGET
        target_center = (target_min + target_max) / 2
        
        # エラー計算
        if target_min <= spine_angle <= target_max:
            error = 0
        else:
            error = min(abs(spine_angle - target_min), abs(spine_angle - target_max))
            
        # スコア計算（より寛容な評価）
        if error == 0:
            score = 100
        elif error <= 10:
            score = 100 - error * 1  # より緩やかな減点
        else:
            score = max(70, 90 - (error - 10) * 1)  # 最低70点保証
            
        # コメント生成
        if score >= 90:
            comment = f"背筋は適切に伸びています"
        elif spine_angle < target_min:
            comment = f"背筋をもう少し伸ばしてください（現在: {spine_angle:.1f}°）"
        else:
            comment = f"適度な前傾を保ってください"
            
        return spine_angle, target_center, error, score, comment
        
    except Exception as e:
        return None, None, None, 15, f"背中の姿勢評価エラー（最低点数付与）: {e}"


def evaluate_forward_tilt(landmarks, club):
    """前傾角度の評価"""
    try:
        # 肩と腰の座標取得
        left_shoulder = landmarks[11][:2]
        right_shoulder = landmarks[12][:2]
        left_hip = landmarks[23][:2]
        right_hip = landmarks[24][:2]
        
        # 各部位の中点
        shoulder_center = [(left_shoulder[0] + right_shoulder[0]) / 2, (left_shoulder[1] + right_shoulder[1]) / 2]
        hip_center = [(left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2]
        
        # 前傾角度計算（肩→腰ベクトルと鉛直の角度）
        forward_tilt = math.degrees(math.atan2(abs(shoulder_center[0] - hip_center[0]), abs(shoulder_center[1] - hip_center[1])))
        
        target = FORWARD_TILT_TARGETS[club]
        error = abs(forward_tilt - target)
        
        # スコア計算（より寛容な評価）
        if error <= TILT_TOLERANCE:
            score = 100
        elif error <= 15:
            score = 100 - (error - TILT_TOLERANCE) * 1  # より緩やかな減点
        else:
            score = max(70, 90 - (error - 15) * 1)  # 最低70点保証
            
        # コメント生成
        if score >= 90:
            comment = f"前傾角度は最適です（{forward_tilt:.1f}°）"
        elif forward_tilt < target:
            comment = f"前傾角度をあと{target - forward_tilt:.1f}°深くしてください"
        else:
            comment = f"前傾角度をあと{forward_tilt - target:.1f}°浅くしてください"
            
        return forward_tilt, target, error, score, comment
        
    except Exception as e:
        return None, None, None, 15, f"前傾角度評価エラー（最低点数付与）: {e}"


def create_radar_chart(scores, categories):
    """レーダーチャートを作成"""
    fig = go.Figure()
    
    # スタンス幅を最初（上部）に配置するために項目を並び替え
    if "スタンス幅" in categories:
        stance_idx = categories.index("スタンス幅")
        # スタンス幅を最初に移動
        reordered_categories = ["スタンス幅"] + [cat for cat in categories if cat != "スタンス幅"]
        reordered_scores = [scores[stance_idx]] + [scores[i] for i, cat in enumerate(categories) if cat != "スタンス幅"]
    else:
        reordered_categories = categories
        reordered_scores = scores
    
    fig.add_trace(go.Scatterpolar(
        r=reordered_scores,
        theta=reordered_categories,
        fill='toself',
        name='スコア',
        line=dict(color='rgb(0, 150, 200)', width=3),
        fillcolor='rgba(0, 150, 200, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickmode='linear',
                tick0=0,
                dtick=20
            )
        ),
        showlegend=False,
        title={
            'text': "アドレス診断レーダーチャート",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        font=dict(size=14)
    )
    
    return fig


def draw_overlay(image, landmarks, results, ball_pos=None):
    """画像にオーバーレイを描画"""
    overlay_image = image.copy()
    height, width = image.shape[:2]
    
    # ランドマークを描画
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            overlay_image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
        )
    
    # ボール位置を描画
    if ball_pos:
        cv2.circle(overlay_image, (int(ball_pos[0] * width), int(ball_pos[1] * height)), 10, (255, 255, 0), -1)
        cv2.circle(overlay_image, (int(ball_pos[0] * width), int(ball_pos[1] * height)), 12, (0, 0, 0), 2)
    
    # 基準線を描画
    if landmarks is not None:
        # 足幅線
        left_ankle = landmarks[27]
        right_ankle = landmarks[28]
        cv2.line(overlay_image, 
                (int(left_ankle[0] * width), int(left_ankle[1] * height)),
                (int(right_ankle[0] * width), int(right_ankle[1] * height)),
                (255, 0, 255), 3)
        
        # 肩線
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        cv2.line(overlay_image,
                (int(left_shoulder[0] * width), int(left_shoulder[1] * height)),
                (int(right_shoulder[0] * width), int(right_shoulder[1] * height)),
                (0, 255, 255), 3)
    
    return overlay_image


def save_session_data(total_score, evaluations, club, shooting_direction, is_left_handed):
    """セッションデータを保存"""
    if 'session_history' not in st.session_state:
        st.session_state.session_history = []
    
    session_data = {
        'timestamp': datetime.now().isoformat(),
        'total_score': total_score,
        'club': club,
        'shooting_direction': shooting_direction,
        'is_left_handed': is_left_handed,
        'evaluations': {k: v[3] if len(v) > 3 else 0 for k, v in evaluations.items()}  # スコアのみ保存
    }
    
    st.session_state.session_history.append(session_data)
    
    # 最新10件まで保持
    if len(st.session_state.session_history) > 10:
        st.session_state.session_history = st.session_state.session_history[-10:]


def display_session_history():
    """セッション履歴の表示"""
    if 'session_history' not in st.session_state or not st.session_state.session_history:
        st.info("📝 まだ診断履歴がありません。診断を実行すると履歴が表示されます。")
        return
    
    st.subheader("📊 診断履歴")
    
    # 履歴データの整理
    history_data = []
    for i, session in enumerate(reversed(st.session_state.session_history)):
        dt = datetime.fromisoformat(session['timestamp'])
        history_data.append({
            '診断回数': f"#{len(st.session_state.session_history) - i}",
            '日時': dt.strftime('%m/%d %H:%M'),
            'クラブ': session['club'],
            '撮影方向': session['shooting_direction'],
            '総合スコア': f"{session['total_score']:.0f}点",
            '利き手': '左打ち' if session['is_left_handed'] else '右打ち'
        })
    
    df_history = pd.DataFrame(history_data)
    st.dataframe(df_history, use_container_width=True)
    
    # スコア推移グラフ
    if len(st.session_state.session_history) > 1:
        st.subheader("📈 スコア推移")
        
        scores = [session['total_score'] for session in st.session_state.session_history]
        timestamps = [datetime.fromisoformat(session['timestamp']) for session in st.session_state.session_history]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(1, len(scores) + 1)),
            y=scores,
            mode='lines+markers',
            name='総合スコア',
            line=dict(color='rgb(0, 150, 200)', width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title="診断回数別スコア推移",
            xaxis_title="診断回数",
            yaxis_title="スコア",
            yaxis=dict(range=[0, 100]),
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)


def main():
    # タイトル
    st.title("⛳ ゴルフアドレス診断ツール")
    
    # タブの作成
    tab1, tab2 = st.tabs(["🎯 診断", "📊 履歴"])
    
    with tab2:
        display_session_history()
    
    with tab1:
        st.markdown("---")
        
        # 使い方の説明（拡張）
        with st.expander("📖 使い方・ヒント", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📷 写真撮影のコツ")
                st.write("""
                - **全身が写るように**: 頭からつま先まで画面内に収める
                - **明るい場所で**: 屋外や明るい室内での撮影を推奨
                - **正面/後方から**: クラブと体の位置関係が分かるアングル
                - **背景はシンプルに**: 人物が識別しやすい背景を選ぶ
                """)
                
            with col2:
                st.markdown("### 🎯 診断の進め方")
                st.write("""
                1. **画像をアップロード**: アドレス時の写真を選択
                2. **設定を選択**: 右打ち/左打ち、撮影方向、使用クラブ
                3. **ボール位置を確認**: 自動検出結果を手動で調整可能
                4. **結果を確認**: スコア、レーダーチャート、改善点をチェック
                """)
        
        # サイドバー設定
        with st.sidebar:
            st.header("⚙️ 設定")
            
            # 基本設定セクション
            with st.container():
                st.subheader("🏌️ 基本設定")
                is_left_handed = st.radio("打ち手", ["右打ち", "左打ち"], help="あなたの利き手を選択してください") == "左打ち"
                shooting_direction = st.radio("撮影方向", ["正面", "後方"], help="写真を撮影した角度を選択してください")
            
            st.divider()
            
            # クラブ選択セクション
            with st.container():
                st.subheader("🏌️‍♂️ 使用クラブ")
                club_type = st.selectbox("クラブタイプ", ["ドライバー", "フェアウェイウッド", "ユーティリティ", "アイアン", "ウェッジ"], 
                                       help="使用したクラブの種類を選択してください")
                
                if club_type == "アイアン":
                    iron_number = st.selectbox("アイアン番手", ["3I", "4I", "5I", "6I", "7I", "8I", "9I"])
                    club = iron_number
                elif club_type == "ウェッジ":
                    wedge_type = st.selectbox("ウェッジタイプ", ["PW", "AW", "SW", "LW"])
                    club = wedge_type
                else:
                    if club_type == "フェアウェイウッド":
                        fw_number = st.selectbox("フェアウェイウッド", ["3W", "4W", "5W", "7W"])
                        club = fw_number
                    elif club_type == "ユーティリティ":
                        ut_number = st.selectbox("ユーティリティ", ["3UT", "4UT", "5UT", "6UT"])
                        club = ut_number
                    elif club_type == "ドライバー":
                        club = "Driver"
        
        
        # デフォルト重みを設定
        weights = {
            "ボール位置": 1.0,
            "スタンス幅": 1.0,
            "手の位置": 1.0,
            "頭の位置": 1.0,
            "体の向き（肩、腰、かかと）": 1.0,
            "背筋の曲がり具合": 1.0,
            "前傾角度": 1.0,
            "重心配分": 1.0,
            "重心の位置": 1.0
        }
    
    # カスタムCSSでファイルアップローダーのボタンテキストを変更
    st.markdown("""
    <style>
    /* ファイルアップローダーのボタンテキストを完全に置換 */
    [data-testid="stFileUploader"] button p {
        display: none !important;
    }
    
    [data-testid="stFileUploader"] button span {
        display: none !important;
    }
    
    [data-testid="stFileUploader"] button {
        position: relative;
    }
    
    [data-testid="stFileUploader"] button::before {
        content: "ファイルを選択" !important;
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        font-size: 14px;
        font-weight: 400;
        color: inherit;
    }
    
    /* すべてのテキストを非表示にする */
    [data-testid="stFileUploader"] button * {
        opacity: 0 !important;
    }
    
    /* 新しいテキストのみ表示 */
    [data-testid="stFileUploader"] button::before {
        opacity: 1 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # 画像アップロード
    uploaded_file = st.file_uploader("📷 アドレス写真をアップロード", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # 画像読み込み
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        # ランドマーク抽出
        with st.spinner("姿勢を解析中..."):
            landmarks, results = extract_landmarks(image_cv)
        
        if landmarks is not None:
            # 成功メッセージと進捗表示
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("✅ 姿勢検出完了 - ボール位置を解析中...")
            progress_bar.progress(33)
            
            # ボール位置の手動指定（オプション）
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("📸 解析画像")
                
                # ボール位置の処理（正面撮影時のみ）
                ball_pos = None
                detection_info = {"detected": False, "method": "not_needed", "confidence": 0}
                
                if shooting_direction == "正面":
                    # セッションステートでボール位置を管理
                    if "ball_position" not in st.session_state:
                        st.session_state.ball_position = None
                    
                    # ボール位置の自動検出（初回または位置がない場合）
                    if st.session_state.ball_position is None:
                        ball_pos, detection_info = detect_ball_automatically(image_cv, landmarks)
                        st.session_state.ball_position = ball_pos
                        st.session_state.detection_info = detection_info
                    else:
                        ball_pos = st.session_state.ball_position
                        detection_info = st.session_state.detection_info
                    
                    # 検出情報を表示
                    if detection_info["detected"]:
                        confidence = detection_info.get("confidence", 0)
                        method = detection_info.get("method", "unknown")
                        
                        if confidence >= 80:
                            st.success(f"✅ ボールを高精度で検出しました（信頼度: {confidence}%）")
                        elif confidence >= 60:
                            st.info(f"🎯 ボールを検出しました（信頼度: {confidence}%）。必要に応じて手動調整してください。")
                        else:
                            st.warning(f"⚠️ ボール検出の信頼度が低いです（{confidence}%）。手動で調整することをお勧めします。")
                            
                        # 検出方法の詳細
                        with st.expander("🔍 検出詳細"):
                            st.write(f"検出方法: {method}")
                            st.write(f"信頼度: {confidence}%")
                            if "radius" in detection_info:
                                st.write(f"検出半径: {detection_info['radius']}px")
                            if "brightness" in detection_info:
                                st.write(f"明度: {detection_info['brightness']}")
                    else:
                        st.error("❌ ボール自動検出に失敗しました。下の画像でボール位置をクリックして指定してください")
                else:
                    # 後方撮影時は説明のみ表示
                    st.info("📍 後方撮影では姿勢バランスと体の向きを重点的に診断します")
                
                # 手動調整オプション（正面撮影時のみ）
                manual_adjustment = False
                if shooting_direction == "正面":
                    if not detection_info["detected"] or st.checkbox("ボール位置を手動調整"):
                        manual_adjustment = True
                        if not detection_info["detected"]:
                            st.info("👆 下の画像でボール位置をクリックしてください")
                        else:
                            st.info("下の画像でボール位置をクリックして調整できます")
                
                # オーバーレイ画像作成
                if shooting_direction == "正面":
                    display_ball_pos = st.session_state.ball_position if st.session_state.ball_position is not None else ball_pos
                else:
                    display_ball_pos = None  # 後方撮影時はボール位置表示なし
                    
                overlay_img = draw_overlay(image_cv, landmarks, results, display_ball_pos)
                overlay_rgb = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB)
                
                # 手動調整時はクリック可能な画像を表示（正面撮影時のみ）
                if manual_adjustment and shooting_direction == "正面":
                    st.info("🖱️ 画像をクリックしてボール位置を指定してください")
                    coordinates = streamlit_image_coordinates(
                        overlay_rgb,
                        width=600,
                        key="image_click"
                    )
                    
                    # クリック座標を取得してボール位置を更新
                    if coordinates is not None and "x" in coordinates and "y" in coordinates:
                        # 画像サイズを取得
                        img_height, img_width = overlay_rgb.shape[:2]
                        # 表示サイズ（600px）に対する実際の画像サイズの比率を計算
                        display_width = 600
                        scale_factor = img_width / display_width
                        display_height = int(img_height / scale_factor)
                        
                        # クリック座標を正規化（0-1の範囲）
                        new_ball_pos = [coordinates["x"] / display_width, coordinates["y"] / display_height]
                        
                        # ボール位置の変更を検出
                        old_pos = st.session_state.ball_position
                        distance = abs(old_pos[0] - new_ball_pos[0]) + abs(old_pos[1] - new_ball_pos[1])
                        
                        if distance > 0.02:  # 2%以上変化した場合のみ更新
                            st.session_state.ball_position = new_ball_pos
                            st.success(f"ボール位置を更新しました")
                            st.rerun()  # ページを再描画して総合スコアを更新
                
                # 通常の画像表示
                if not manual_adjustment:
                    caption = "姿勢解析結果（後方撮影）" if shooting_direction == "後方" else "姿勢解析結果（正面撮影）"
                    st.image(overlay_rgb, caption=caption, width=600)
            
            with col2:
                # 進捗更新
                status_text.text("🔍 アドレス姿勢を分析中...")
                progress_bar.progress(66)
                
                # 撮影方向に応じた評価項目の選択
                evaluations = {}
                
                # 撮影方向に応じた評価項目の選択
                if shooting_direction == "後方":
                    # 後方撮影時の5つのチェック項目のみ
                    # 1. 体の向き（肩、腰、かかと）
                    eval_result = evaluate_body_alignment(landmarks, is_left_handed)
                    evaluations["体の向き（肩、腰、かかと）"] = eval_result
                    
                    # 2. 前傾角度
                    eval_result = evaluate_forward_tilt(landmarks, club)
                    evaluations["前傾角度"] = eval_result
                    
                    # 3. 背筋の曲がり具合
                    eval_result = evaluate_spine_posture(landmarks)
                    evaluations["背筋の曲がり具合"] = eval_result
                    
                    # 4. 重心の位置
                    eval_result = evaluate_weight_distribution(landmarks, club, is_left_handed, shooting_direction)
                    evaluations["重心の位置"] = eval_result
                    
                    # 5. 手の位置
                    eval_result = evaluate_hand_position(landmarks, club, shooting_direction, is_left_handed)
                    evaluations["手の位置"] = eval_result
                else:
                    # 正面撮影時の項目
                    # 最新のボール位置を確実に取得
                    current_ball_pos = st.session_state.ball_position if st.session_state.ball_position is not None else ball_pos
                    
                    # 1. ボール位置
                    eval_result = evaluate_ball_position(landmarks, current_ball_pos, club, is_left_handed)
                    evaluations["ボール位置"] = eval_result
                    
                    # 2. スタンス幅
                    eval_result = evaluate_stance_width(landmarks, club, is_left_handed)
                    evaluations["スタンス幅"] = eval_result
                    
                    # 3. 手の位置
                    eval_result = evaluate_hand_position(landmarks, club, shooting_direction, is_left_handed)
                    evaluations["手の位置"] = eval_result
                    
                    # 4. 頭の位置
                    eval_result = evaluate_head_position(landmarks, club, is_left_handed)
                    evaluations["頭の位置"] = eval_result
                    
                    # 5. 重心配分
                    eval_result = evaluate_weight_distribution(landmarks, club, is_left_handed, shooting_direction)
                    evaluations["重心配分"] = eval_result
                
                # 総合スコア計算（エラー項目も含める）
                valid_scores = []
                valid_weights = []
                valid_categories = []
                
                for category, (_, _, _, score, _) in evaluations.items():
                    if score is not None and score >= 0:  # 有効なスコアのみ
                        valid_scores.append(score)
                        valid_weights.append(weights.get(category, 1.0))
                        valid_categories.append(category)
                
                if valid_scores:
                    total_score = np.average(valid_scores, weights=valid_weights)
                    total_score = max(0, min(100, total_score))  # 0-100の範囲に制限
                else:
                    total_score = 0
                
                # 進捗完了
                status_text.text("✅ 分析完了！")
                progress_bar.progress(100)
                
                # 総合スコア表示
                st.markdown("### 📊 総合スコア")
                
                # スコアに応じた色とメッセージ
                if total_score >= 80:
                    score_color = "🟢"
                    score_message = "🎯 素晴らしいアドレスです！"
                    message_type = "success"
                else:
                    score_color = "🔴"
                    score_message = ""
                    message_type = "none"
                
                # スコア表示レイアウト
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    # 正方形スコア表示
                    st.markdown(f"""
                    <div style="
                        text-align: center; 
                        padding: 40px; 
                        width: 300px;
                        height: 300px;
                        border-radius: 0px; 
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
                        margin: 20px auto;
                        display: flex;
                        flex-direction: column;
                        justify-content: center;
                        align-items: center;
                    ">
                        <h3 style="margin: 0; font-weight: 300; opacity: 0.9; font-size: 1.2em;">100点満点中</h3>
                        <h1 style="
                            margin: 20px 0 0 0; 
                            font-size: 5em; 
                            font-weight: bold;
                            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
                            line-height: 1;
                        ">{total_score:.0f}点</h1>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # 評価項目数の表示
                    if valid_categories:
                        st.caption(f"評価項目数: {len(valid_categories)}項目")
                    
                    # メッセージ表示
                    if message_type == "success":
                        st.success(score_message)
                    
                    # リアルタイムフィードバック: スコア変化の傾向
                    if 'previous_score' in st.session_state:
                        score_diff = total_score - st.session_state.previous_score
                        if abs(score_diff) > 2:  # 2点以上変化した場合
                            if score_diff > 0:
                                st.info(f"📈 前回より{score_diff:.0f}点向上しました！")
                            else:
                                st.warning(f"📉 前回より{abs(score_diff):.0f}点低下しました")
                    
                    # 現在のスコアを保存
                    st.session_state.previous_score = total_score
                    
                    # セッションデータを履歴に保存
                    save_session_data(total_score, evaluations, club, shooting_direction, is_left_handed)
            
            # 詳細結果表示
            st.subheader("📈 詳細診断結果")
            
            # データフレーム作成（優先度付き表示）
            df_data = []
            categories = []
            scores = []
            priority_items = []  # 優先改善項目
            
            for category, (measured, target, error, score, comment) in evaluations.items():
                if score is not None and score >= 0:
                    # スコアに応じた優先度判定
                    if score < 70:
                        priority = "🔴 要改善"
                        priority_items.append((category, score, comment))
                    elif score < 85:
                        priority = "🟡 要注意"
                    else:
                        priority = "🟢 良好"
                    
                    df_data.append({
                        "項目": category,
                        "スコア": f"{score:.0f}点",
                        "優先度": priority,
                        "アドバイス": comment
                    })
                    categories.append(category)
                    scores.append(score)
            
            df = pd.DataFrame(df_data)
            
            # 優先改善項目がある場合は先に表示
            if priority_items:
                st.markdown("### 🎯 重点改善項目")
                for item, score, comment in sorted(priority_items, key=lambda x: x[1]):
                    with st.expander(f"❗ {item} ({score:.0f}点)", expanded=True):
                        st.warning(comment)
                        
                        # 改善のための具体的なヒント
                        if "ボール位置" in item:
                            st.info("💡 **改善ヒント**: ボール位置は番手によって大きく変わります。ドライバーは左足寄り、アイアンは中央寄りが基本です。")
                        elif "スタンス" in item:
                            st.info("💡 **改善ヒント**: スタンス幅は肩幅を基準に調整します。ドライバーは広め、ショートアイアンは狭めが基本です。")
                        elif "前傾" in item:
                            st.info("💡 **改善ヒント**: 前傾角度は股関節から曲げるのがポイント。背中を丸めずに、お尻を後ろに突き出すイメージです。")
                
                st.markdown("---")
            
            # 全体結果テーブル
            st.dataframe(df, use_container_width=True)
            
            # レーダーチャート
            if scores:
                st.subheader("🎯 レーダーチャート")
                radar_fig = create_radar_chart(scores, categories)
                st.plotly_chart(radar_fig)
            
            # ダウンロード機能
            st.subheader("💾 結果のダウンロード")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # CSV ダウンロード
                csv = df.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="📊 CSV形式でダウンロード",
                    data=csv,
                    file_name=f"golf_address_analysis_{club}.csv",
                    mime="text/csv"
                )
            
            with col2:
                # レーダーチャート PNG ダウンロード
                if scores:
                    try:
                        img_bytes = radar_fig.to_image(format="png", width=800, height=600)
                        st.download_button(
                            label="📈 レーダーチャートをダウンロード",
                            data=img_bytes,
                            file_name=f"golf_radar_chart_{club}.png",
                            mime="image/png"
                        )
                    except:
                        st.info("レーダーチャートのPNGダウンロードは利用できません（kaleidoパッケージが必要）")
            
        else:
            st.error("❌ 姿勢を検出できませんでした。以下を確認してください：")
            st.write("- 体全体（頭から足まで）が写っているか")
            st.write("- 画像が明るく、はっきりしているか")
            st.write("- 被写体が画像の中央に位置しているか")
            st.write("- 背景がシンプルで、人物が識別しやすいか")


if __name__ == "__main__":
    main()