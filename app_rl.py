from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
import os
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
import tempfile
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_bcrypt import Bcrypt  
from flask_login import current_user
from datetime import datetime  # 日時情報を取得するために追加
import pytz
from flask_migrate import Migrate  # Migrateクラスのインポートを追加

# アプリケーションファクトリ関数
def create_app():
    app = Flask(__name__)
    app.template_folder = 'templates'

    # config.pyから設定を読み込む
    app.config.from_pyfile('config.py')

    # データベースの設定と初期化
    db = SQLAlchemy(app)
    migrate = Migrate(app, db)

    # ユーザーモデルの作成
    class User(db.Model, UserMixin):
        id = db.Column(db.Integer, primary_key=True)
        username = db.Column(db.String(80), unique=True, nullable=False)
        password = db.Column(db.String(120), nullable=False)

    # ログインマネージャーの設定と初期化
    login_manager = LoginManager()
    login_manager.login_view = 'login'
    login_manager.init_app(app)

    # Bcryptの設定と初期化
    bcrypt = Bcrypt(app) 

    # user_loader デコレータを使用してユーザーロード関数を定義
    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))

    # ユーザー登録
    @app.route('/register', methods=['GET', 'POST'])
    def register():
        if request.method == 'POST':
            username = request.form.get('username')
            password = request.form.get('password')
            
            # パスワードをハッシュ化して保存
            hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
            
            # ユーザーをデータベースに追加
            user = User(username=username, password=hashed_password)
            db.session.add(user)
            db.session.commit()
            
            flash('アカウントが作成されました！', 'success')
            return redirect(url_for('login'))
        return render_template('register.html')

    # ログイン
    @app.route('/login', methods=['GET', 'POST'])
    def login():
        if request.method == 'POST':
            username = request.form.get('username')
            password = request.form.get('password')
        
            # ユーザー名からユーザーオブジェクトを取得
            user = User.query.filter_by(username=username).first()
        
            # ユーザーが存在し、パスワードが一致する場合
            if user and bcrypt.check_password_hash(user.password, password):
                login_user(user)
                flash('ログインに成功しました！', 'success')
                return redirect(url_for('index_rl'))
            else:
                flash('ログインに失敗しました。ユーザー名とパスワードを確認してください。', 'danger')
    
        return render_template('login.html')

    # ログアウト
    @app.route('/logout')
    @login_required
    def logout():
        logout_user()
        return redirect(url_for('index_rl'))

    # モデルの読み込み
    model_DN = load_model('./models/FW_rl_model.h5')

    # 画像のアップロード先フォルダ
    UPLOAD_FOLDER = 'uploads'
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

    # ランダムなファイル名を生成して保存する関数
    def save_uploaded_file(file):
        temp_dir = tempfile.mkdtemp()
        random_filename = secure_filename(os.urandom(24).hex() + ".jpg")
        file_path = os.path.join(temp_dir, random_filename)
        file.save(file_path)

        # アップロードされたファイルをuploadsフォルダに移動またはコピーする
        destination = os.path.join(app.config['UPLOAD_FOLDER'], random_filename)
        os.rename(file_path, destination)  # ファイルを移動 (または shutil.copy でコピー)

        return random_filename, destination

    # 予測結果をデータベースに保存するためのクラス
    class Prediction(db.Model):
        id = db.Column(db.Integer, primary_key=True)
        user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
        filename = db.Column(db.String(120), nullable=False)
        predicted_age = db.Column(db.String(20), nullable=False)
        confidence = db.Column(db.Float, nullable=False)
        prediction_time = db.Column(db.DateTime, default=datetime.utcnow)  # 予測日時を保存するフィールドを追加

    # 日本のタイムゾーンを取得
    japan_timezone = pytz.timezone('Asia/Tokyo')

    # ルートURLへのリクエストを処理する
    @app.route('/', methods=['GET', 'POST'])
    @login_required  # ログインが必要なページとして修正
    def index_rl():
        uploaded_image = None  # アップロードした画像のランダムなファイル名を格納する変数

        # ユーザーに関連する予測履歴をデータベースから取得
        user_predictions = []

        # プレビュー画像のサイズ
        preview_size = (80, 80)

        # 日本時間を取得
        japan_timezone = pytz.timezone('Asia/Tokyo')

        if request.method == 'POST':
            # アップロードされた画像を取得
            uploaded_file = request.files['file']

            # 画像がアップロードされている場合
            if uploaded_file.filename != '':
                # ファイルを一時フォルダに保存し、ランダムなファイル名を取得
                uploaded_image, file_path = save_uploaded_file(uploaded_file)

                # デバッグメッセージで保存先パスを出力
                print("保存先パス:", uploaded_image)

                # 一時ファイルのパスを使用して画像を読み込み、モデルで推論
                img = load_img(file_path, target_size=(224, 224))

                # 画像を読み込み、モデルで推論
                img = Image.open(uploaded_file)
                img = img.resize((224, 224))  # 画像のサイズをモデルの入力サイズに合わせる
                img_array = np.array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = img_array / 255.0  # 画像データを正規化

                # モデルで推論を行う
                predictions = model_DN.predict(img_array)

                # 予測結果を解釈し、予測年齢と確率を取得
                predicted_class = np.argmax(predictions)  # 最も高い確率をもつクラスのインデックス
                class_labels = ['乳児', '年少', '年中', '年長']  # クラスのラベル

                predicted_age = class_labels[predicted_class]  # 予測された年齢
                confidence = predictions[0][predicted_class] * 100  # 予測の確率をパーセントに変換
                confidence = "{:.2f}".format(confidence)

                # 予測日時を取得
                prediction_time = datetime.now(pytz.utc)

                # 予測結果をデータベースに保存
                if not current_user.is_anonymous:
                    prediction = Prediction(
                        user_id=current_user.id,
                        filename=uploaded_image,
                        predicted_age=predicted_age,
                        confidence=confidence,
                        prediction_time=prediction_time
                    )
                    db.session.add(prediction)
                    db.session.commit()

                # 予測結果を表示するページにリダイレクト
                return redirect(url_for('result', age=predicted_age, confidence=confidence, uploaded_image=uploaded_image))

        # ユーザーがログインしている場合にのみ処理
        if not current_user.is_anonymous:
            # データベースからユーザーの予測結果を取得
            user_predictions = Prediction.query.filter_by(user_id=current_user.id).all()

            # 予測日時を日本時間に変換
            for prediction in user_predictions:
                if prediction.prediction_time:
                    japan_prediction_time = prediction.prediction_time.astimezone(japan_timezone)
                    # 予測日時を prediction オブジェクトに追加
                    prediction.japan_prediction_time = japan_prediction_time

        # ここでテンプレートに渡す変数を定義
        return render_template('index_rl.html', uploaded_image=uploaded_image, user_predictions=user_predictions, preview_size=preview_size)


    # 画像ファイルへのリクエストを処理する
    @app.route('/uploads/<filename>')
    def uploaded_file(filename):
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

    # 予測結果を表示するページを処理する
    @app.route('/result')
    def result():
        predicted_age = request.args.get('age', '予測年齢')
        confidence = request.args.get('confidence', '確率%')
        uploaded_image = request.args.get('uploaded_image', '')  # ランダムなファイル名を受け取る
        return render_template('result.html', uploaded_image=uploaded_image, predicted_age=predicted_age, confidence=confidence)

    # このようにアプリケーションコンテキストを確立する
    with app.app_context():
        db.create_all()

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, port=8000)

