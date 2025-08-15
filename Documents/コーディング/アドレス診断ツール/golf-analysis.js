// ゴルフアドレス診断ツール - JavaScript版

class GolfAddressDiagnosisApp {
    constructor() {
        this.pose = null;
        this.currentImage = null;
        this.landmarks = null;
        this.ballPosition = null;
        
        this.initializeMediaPipe();
        this.setupEventListeners();
        this.setupDragAndDrop();
    }

    initializeMediaPipe() {
        this.pose = new Pose({
            locateFile: (file) => {
                return `https://cdn.jsdelivr.net/npm/@mediapipe/pose@0.5.1675469404/${file}`;
            }
        });

        this.pose.setOptions({
            modelComplexity: 1,
            smoothLandmarks: true,
            enableSegmentation: false,
            smoothSegmentation: true,
            minDetectionConfidence: 0.5,
            minTrackingConfidence: 0.5
        });

        this.pose.onResults(this.onResults.bind(this));
    }

    setupEventListeners() {
        const imageInput = document.getElementById('image-input');
        imageInput.addEventListener('change', this.handleImageUpload.bind(this));
    }

    setupDragAndDrop() {
        const uploadSection = document.getElementById('upload-section');
        
        uploadSection.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadSection.classList.add('drag-over');
        });

        uploadSection.addEventListener('dragleave', () => {
            uploadSection.classList.remove('drag-over');
        });

        uploadSection.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadSection.classList.remove('drag-over');
            
            const files = e.dataTransfer.files;
            if (files.length > 0 && files[0].type.startsWith('image/')) {
                this.processImage(files[0]);
            }
        });
    }

    async handleImageUpload(event) {
        const file = event.target.files[0];
        if (file) {
            await this.processImage(file);
        }
    }

    async processImage(file) {
        // ローディング表示
        this.showLoading();
        
        // 画像を読み込み
        const imageUrl = URL.createObjectURL(file);
        const image = new Image();
        
        image.onload = async () => {
            this.currentImage = image;
            await this.analyzeImage(image);
            URL.revokeObjectURL(imageUrl);
        };
        
        image.src = imageUrl;
    }

    async analyzeImage(image) {
        try {
            // MediaPipeで姿勢解析
            await this.pose.send({ image: image });
        } catch (error) {
            console.error('姿勢解析エラー:', error);
            this.hideLoading();
            alert('姿勢解析に失敗しました。別の画像をお試しください。');
        }
    }

    onResults(results) {
        if (results.poseLandmarks) {
            this.landmarks = results.poseLandmarks;
            this.drawResults(results);
            this.performDiagnosis();
        } else {
            this.hideLoading();
            alert('姿勢を検出できませんでした。全身が写った明るい画像をお試しください。');
        }
    }

    drawResults(results) {
        const canvas = document.getElementById('output-canvas');
        const ctx = canvas.getContext('2d');
        
        // キャンバスサイズを画像に合わせて調整
        canvas.width = this.currentImage.width;
        canvas.height = this.currentImage.height;
        
        // 画像を描画
        ctx.drawImage(this.currentImage, 0, 0);
        
        // ランドマークを描画
        if (results.poseLandmarks) {
            this.drawLandmarks(ctx, results.poseLandmarks);
            this.drawConnections(ctx, results.poseLandmarks);
        }
        
        // ボール位置を描画（正面撮影時のみ）
        const shootingDirection = document.getElementById('shooting-direction').value;
        if (shootingDirection === 'front' && this.ballPosition) {
            this.drawBallPosition(ctx);
        }
        
        // 画像表示
        document.getElementById('image-display').classList.remove('hidden');
    }

    drawLandmarks(ctx, landmarks) {
        ctx.fillStyle = '#00FF00';
        landmarks.forEach(landmark => {
            if (landmark.visibility > 0.5) {
                ctx.beginPath();
                ctx.arc(
                    landmark.x * ctx.canvas.width,
                    landmark.y * ctx.canvas.height,
                    3, 0, 2 * Math.PI
                );
                ctx.fill();
            }
        });
    }

    drawConnections(ctx, landmarks) {
        const connections = [
            [11, 12], [11, 13], [12, 14], [13, 15], [14, 16], // 上半身
            [11, 23], [12, 24], [23, 24], // 胴体
            [23, 25], [24, 26], [25, 27], [26, 28] // 下半身
        ];

        ctx.strokeStyle = '#0000FF';
        ctx.lineWidth = 2;
        
        connections.forEach(([start, end]) => {
            const startPoint = landmarks[start];
            const endPoint = landmarks[end];
            
            if (startPoint.visibility > 0.5 && endPoint.visibility > 0.5) {
                ctx.beginPath();
                ctx.moveTo(
                    startPoint.x * ctx.canvas.width,
                    startPoint.y * ctx.canvas.height
                );
                ctx.lineTo(
                    endPoint.x * ctx.canvas.width,
                    endPoint.y * ctx.canvas.height
                );
                ctx.stroke();
            }
        });
    }

    drawBallPosition(ctx) {
        if (!this.ballPosition) return;
        
        const x = this.ballPosition.x * ctx.canvas.width;
        const y = this.ballPosition.y * ctx.canvas.height;
        
        // ボールの円
        ctx.fillStyle = '#FFFF00';
        ctx.beginPath();
        ctx.arc(x, y, 8, 0, 2 * Math.PI);
        ctx.fill();
        
        // 外枠
        ctx.strokeStyle = '#000000';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(x, y, 8, 0, 2 * Math.PI);
        ctx.stroke();
    }

    performDiagnosis() {
        const club = document.getElementById('club-type').value;
        const isLeftHanded = document.getElementById('handedness').value === 'left';
        const shootingDirection = document.getElementById('shooting-direction').value;
        
        // ボール位置を推定（正面撮影時のみ）
        if (shootingDirection === 'front') {
            this.ballPosition = this.estimateBallPosition();
        }
        
        // 各項目を評価
        const evaluations = this.evaluateAllAspects(club, isLeftHanded, shootingDirection);
        
        // 結果を表示
        this.displayResults(evaluations);
        
        this.hideLoading();
    }

    estimateBallPosition() {
        // 簡易的なボール位置推定（足首の中央前方）
        const leftAnkle = this.landmarks[27];
        const rightAnkle = this.landmarks[28];
        
        return {
            x: (leftAnkle.x + rightAnkle.x) / 2,
            y: Math.min(leftAnkle.y, rightAnkle.y) + 0.1
        };
    }

    evaluateAllAspects(club, isLeftHanded, shootingDirection) {
        const evaluations = {};
        
        if (shootingDirection === 'front') {
            // 正面撮影時の評価項目
            evaluations['ボール位置'] = this.evaluateBallPosition(club, isLeftHanded);
            evaluations['スタンス幅'] = this.evaluateStanceWidth(club);
            evaluations['頭の位置'] = this.evaluateHeadPosition(club, isLeftHanded);
            evaluations['手の位置'] = this.evaluateHandPosition(club, isLeftHanded);
            evaluations['重心配分'] = this.evaluateWeightDistribution(club, isLeftHanded);
        } else {
            // 後方撮影時の評価項目
            evaluations['体の向き'] = this.evaluateBodyAlignment(isLeftHanded);
            evaluations['前傾角度'] = this.evaluateForwardTilt(club);
            evaluations['背筋の姿勢'] = this.evaluateSpinePosture();
            evaluations['手の位置'] = this.evaluateHandPosition(club, isLeftHanded);
            evaluations['重心の位置'] = this.evaluateWeightPosition(isLeftHanded);
        }
        
        return evaluations;
    }

    evaluateBallPosition(club, isLeftHanded) {
        if (!this.ballPosition) {
            return { score: 50, comment: 'ボール位置を検出できませんでした' };
        }
        
        const ballTargets = {
            'Driver': 0.88, '3W': 0.82, '5W': 0.78, '7W': 0.76,
            '3UT': 0.75, '5UT': 0.70, '7I': 0.52, '9I': 0.48,
            'PW': 0.45, 'SW': 0.40
        };
        
        const leftAnkle = this.landmarks[27];
        const rightAnkle = this.landmarks[28];
        
        if (isLeftHanded) {
            [leftAnkle, rightAnkle] = [rightAnkle, leftAnkle];
        }
        
        const footWidth = Math.abs(leftAnkle.x - rightAnkle.x);
        const ballRelative = (this.ballPosition.x - rightAnkle.x) / footWidth;
        
        const target = ballTargets[club] || 0.5;
        const error = Math.abs(ballRelative - target);
        
        let score = 100;
        if (error > 0.06) {
            score = Math.max(60, 100 - (error - 0.06) * 200);
        }
        
        let comment = '';
        if (score >= 90) {
            comment = `ボール位置は最適です（目標: ${target.toFixed(2)}, 実測: ${ballRelative.toFixed(2)}）`;
        } else if (ballRelative < target) {
            comment = `ボール位置を左足側に移動してください`;
        } else {
            comment = `ボール位置を右足側に移動してください`;
        }
        
        return { score: Math.round(score), comment };
    }

    evaluateStanceWidth(club) {
        const leftAnkle = this.landmarks[27];
        const rightAnkle = this.landmarks[28];
        const leftShoulder = this.landmarks[11];
        const rightShoulder = this.landmarks[12];
        
        const stanceWidth = this.calculateDistance(leftAnkle, rightAnkle);
        const shoulderWidth = this.calculateDistance(leftShoulder, rightShoulder);
        
        const stanceRatio = stanceWidth / shoulderWidth;
        
        const targets = {
            'Driver': [1.20, 1.40],
            'default': [0.95, 1.20]
        };
        
        const targetRange = targets[club] || targets['default'];
        const targetCenter = (targetRange[0] + targetRange[1]) / 2;
        
        let error = 0;
        if (stanceRatio < targetRange[0]) {
            error = targetRange[0] - stanceRatio;
        } else if (stanceRatio > targetRange[1]) {
            error = stanceRatio - targetRange[1];
        }
        
        let score = 100;
        if (error > 0) {
            score = Math.max(65, 100 - error * 100);
        }
        
        let comment = '';
        if (score >= 90) {
            comment = `スタンス幅は最適です（比率: ${stanceRatio.toFixed(2)}）`;
        } else if (stanceRatio < targetRange[0]) {
            comment = `スタンスをもう少し広くしてください`;
        } else {
            comment = `スタンスをもう少し狭くしてください`;
        }
        
        return { score: Math.round(score), comment };
    }

    evaluateHeadPosition(club, isLeftHanded) {
        const nose = this.landmarks[0];
        const leftAnkle = this.landmarks[27];
        const rightAnkle = this.landmarks[28];
        
        if (isLeftHanded) {
            [leftAnkle, rightAnkle] = [rightAnkle, leftAnkle];
        }
        
        const footWidth = Math.abs(leftAnkle.x - rightAnkle.x);
        const headRelative = (nose.x - rightAnkle.x) / footWidth;
        
        const target = club === 'Driver' ? 0.52 : 0.50;
        const error = Math.abs(headRelative - target);
        
        let score = 100;
        if (error > 0.05) {
            score = Math.max(65, 100 - (error - 0.05) * 150);
        }
        
        let comment = '';
        if (score >= 90) {
            comment = '頭の位置は最適です';
        } else if (headRelative < target) {
            comment = '頭をもう少し右に寄せてください';
        } else {
            comment = '頭をもう少し左に寄せてください';
        }
        
        return { score: Math.round(score), comment };
    }

    evaluateHandPosition(club, isLeftHanded) {
        // 簡易的な手の位置評価
        const leftWrist = this.landmarks[15];
        const rightWrist = this.landmarks[16];
        const leftHip = this.landmarks[23];
        const rightHip = this.landmarks[24];
        
        const handCenter = {
            x: (leftWrist.x + rightWrist.x) / 2,
            y: (leftWrist.y + rightWrist.y) / 2
        };
        
        const hipCenter = {
            x: (leftHip.x + rightHip.x) / 2,
            y: (leftHip.y + rightHip.y) / 2
        };
        
        const distance = this.calculateDistance(handCenter, hipCenter);
        
        // 距離ベースの簡易評価
        let score = 85; // 基本スコア
        if (distance < 0.1) score = 90;
        if (distance > 0.2) score = 70;
        
        return {
            score: Math.round(score),
            comment: '手の位置は概ね良好です'
        };
    }

    evaluateWeightDistribution(club, isLeftHanded) {
        // 簡易的な重心配分評価
        const leftHip = this.landmarks[23];
        const rightHip = this.landmarks[24];
        const leftShoulder = this.landmarks[11];
        const rightShoulder = this.landmarks[12];
        
        const hipCenter = (leftHip.x + rightHip.x) / 2;
        const shoulderCenter = (leftShoulder.x + rightShoulder.x) / 2;
        
        const weightShift = shoulderCenter - hipCenter;
        
        let score = 80; // 基本スコア
        if (Math.abs(weightShift) < 0.02) score = 95;
        if (Math.abs(weightShift) > 0.05) score = 70;
        
        return {
            score: Math.round(score),
            comment: '重心配分は概ね良好です'
        };
    }

    evaluateBodyAlignment(isLeftHanded) {
        // 体の向きの評価（簡易版）
        const leftShoulder = this.landmarks[11];
        const rightShoulder = this.landmarks[12];
        const leftHip = this.landmarks[23];
        const rightHip = this.landmarks[24];
        
        const shoulderAngle = Math.atan2(leftShoulder.y - rightShoulder.y, leftShoulder.x - rightShoulder.x);
        const hipAngle = Math.atan2(leftHip.y - rightHip.y, leftHip.x - rightHip.x);
        
        const angleDiff = Math.abs(shoulderAngle - hipAngle) * 180 / Math.PI;
        
        let score = 100;
        if (angleDiff > 5) {
            score = Math.max(70, 100 - (angleDiff - 5) * 3);
        }
        
        return {
            score: Math.round(score),
            comment: score >= 90 ? '体の向きは適切です' : '肩と腰の向きを揃えてください'
        };
    }

    evaluateForwardTilt(club) {
        // 前傾角度の評価（簡易版）
        const leftShoulder = this.landmarks[11];
        const rightShoulder = this.landmarks[12];
        const leftHip = this.landmarks[23];
        const rightHip = this.landmarks[24];
        
        const shoulderCenter = {
            x: (leftShoulder.x + rightShoulder.x) / 2,
            y: (leftShoulder.y + rightShoulder.y) / 2
        };
        
        const hipCenter = {
            x: (leftHip.x + rightHip.x) / 2,
            y: (leftHip.y + rightHip.y) / 2
        };
        
        const forwardTilt = Math.atan2(
            Math.abs(shoulderCenter.x - hipCenter.x),
            Math.abs(shoulderCenter.y - hipCenter.y)
        ) * 180 / Math.PI;
        
        const targets = {
            'Driver': 30, '3W': 32, '5W': 34, '7W': 35,
            '3UT': 35, '5UT': 37, '7I': 41, '9I': 43,
            'PW': 45, 'SW': 45
        };
        
        const target = targets[club] || 35;
        const error = Math.abs(forwardTilt - target);
        
        let score = 100;
        if (error > 5) {
            score = Math.max(70, 100 - (error - 5) * 2);
        }
        
        return {
            score: Math.round(score),
            comment: score >= 90 ? `前傾角度は最適です（${forwardTilt.toFixed(1)}°）` : 
                     forwardTilt < target ? '前傾をもう少し深くしてください' : '前傾をもう少し浅くしてください'
        };
    }

    evaluateSpinePosture() {
        // 背筋の姿勢評価（簡易版）
        const nose = this.landmarks[0];
        const leftShoulder = this.landmarks[11];
        const rightShoulder = this.landmarks[12];
        const leftHip = this.landmarks[23];
        const rightHip = this.landmarks[24];
        
        const shoulderCenter = {
            x: (leftShoulder.x + rightShoulder.x) / 2,
            y: (leftShoulder.y + rightShoulder.y) / 2
        };
        
        const hipCenter = {
            x: (leftHip.x + rightHip.x) / 2,
            y: (leftHip.y + rightHip.y) / 2
        };
        
        // 簡易的な背筋の曲がり評価
        const spineAlignment = this.calculateDistance(nose, shoulderCenter) + 
                              this.calculateDistance(shoulderCenter, hipCenter);
        
        let score = 85; // 基本スコア
        if (spineAlignment < 0.3) score = 95;
        if (spineAlignment > 0.5) score = 70;
        
        return {
            score: Math.round(score),
            comment: score >= 90 ? '背筋は適切に伸びています' : '背筋をもう少し伸ばしてください'
        };
    }

    evaluateWeightPosition(isLeftHanded) {
        // 重心位置の評価（簡易版）
        const leftShoulder = this.landmarks[11];
        const rightShoulder = this.landmarks[12];
        const leftAnkle = this.landmarks[27];
        const rightAnkle = this.landmarks[28];
        
        const shoulderCenter = (leftShoulder.x + rightShoulder.x) / 2;
        const ankleCenter = (leftAnkle.x + rightAnkle.x) / 2;
        
        const deviation = Math.abs(shoulderCenter - ankleCenter);
        
        let score = 100;
        if (deviation > 0.05) {
            score = Math.max(75, 100 - (deviation - 0.05) * 200);
        }
        
        return {
            score: Math.round(score),
            comment: score >= 90 ? '重心位置は理想的です' : '重心位置を調整してください'
        };
    }

    calculateDistance(point1, point2) {
        return Math.sqrt(
            Math.pow(point1.x - point2.x, 2) + Math.pow(point1.y - point2.y, 2)
        );
    }

    displayResults(evaluations) {
        // 総合スコア計算
        const scores = Object.values(evaluations).map(e => e.score);
        const totalScore = Math.round(scores.reduce((a, b) => a + b, 0) / scores.length);
        
        // 総合スコア表示
        document.getElementById('total-score').textContent = totalScore;
        
        // 評価結果表示
        const resultsContainer = document.getElementById('evaluation-results');
        resultsContainer.innerHTML = '';
        
        Object.entries(evaluations).forEach(([category, evaluation]) => {
            const item = document.createElement('div');
            item.className = 'evaluation-item';
            
            let scoreClass = 'score-excellent';
            if (evaluation.score < 85) scoreClass = 'score-good';
            if (evaluation.score < 70) scoreClass = 'score-poor';
            
            item.innerHTML = `
                <h3>${category}</h3>
                <div class="evaluation-score ${scoreClass}">${evaluation.score}点</div>
                <p>${evaluation.comment}</p>
            `;
            
            resultsContainer.appendChild(item);
        });
        
        // レーダーチャート表示
        this.displayRadarChart(evaluations);
        
        // 結果セクション表示
        document.getElementById('results-section').classList.remove('hidden');
    }

    displayRadarChart(evaluations) {
        const categories = Object.keys(evaluations);
        const scores = Object.values(evaluations).map(e => e.score);
        
        const data = [{
            type: 'scatterpolar',
            r: scores,
            theta: categories,
            fill: 'toself',
            name: 'スコア',
            line: { color: '#667eea' },
            fillcolor: 'rgba(102, 126, 234, 0.3)'
        }];
        
        const layout = {
            polar: {
                radialaxis: {
                    visible: true,
                    range: [0, 100]
                }
            },
            showlegend: false,
            title: {
                text: 'アドレス診断レーダーチャート',
                font: { size: 18 }
            }
        };
        
        Plotly.newPlot('radar-plot', data, layout);
    }

    showLoading() {
        document.getElementById('loading').classList.remove('hidden');
        document.getElementById('results-section').classList.add('hidden');
    }

    hideLoading() {
        document.getElementById('loading').classList.add('hidden');
    }
}

// アプリ初期化
document.addEventListener('DOMContentLoaded', () => {
    new GolfAddressDiagnosisApp();
});