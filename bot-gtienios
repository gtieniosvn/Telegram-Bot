import re
import json
import os
import threading
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, CommandHandler, ContextTypes
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# ==================== CẤU HÌNH ====================
TOKEN = "8769658167:AAFsbd2IHEisKtku77dRGkfaYCkfF4NJLjU"  # Thay token của bạn
ADMIN_CHAT_IDS = {8666979500}  # Thay chat_id admin của bạn
DASHBOARD_PORT = 5000
# =================================================

DATA_FILE = "bad_words.json"
MODEL_FILE = "toxic_model.pkl"
VECTOR_FILE = "vectorizer.pkl"
LOG_FILE = "violations.json"

# ------------------ KHỞI TẠO FLASK ------------------
app_flask = Flask(__name__)
app_flask.config['SECRET_KEY'] = '91a4d69d92ca457ac8918f082e9357f3b264d84256f718ee93f9539e43399775'
socketio = SocketIO(app_flask, cors_allowed_origins="*")

violations = []

def load_logs():
    global violations
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            violations = json.load(f)

def save_logs():
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(violations, f, ensure_ascii=False, indent=2)

load_logs()

# ------------------ AI TRAIN ------------------
train_texts = [
    ("mày ngu như chó", 1), ("đồ khốn nạn", 1), ("thằng lồn", 1), ("cút mẹ mày đi", 1),
    ("đĩ thối", 1), ("chết mẹ mày", 1), ("ngu si", 1), ("óc chó", 1), ("mất dạy", 1),
    ("súc vật", 1), ("cặc", 1), ("buồi", 1), ("đụ mẹ", 1), ("vãi lồn", 1),
    ("chào bạn", 0), ("hôm nay đẹp trời", 0), ("cảm ơn", 0), ("giúp mình với", 0),
    ("bạn khỏe không", 0), ("học bài chưa", 0), ("ăn cơm chưa", 0), ("vui quá", 0),
]
X_train = [t[0] for t in train_texts]
y_train = [t[1] for t in train_texts]

if not os.path.exists(MODEL_FILE):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    X_vec = vectorizer.fit_transform(X_train)
    clf = MultinomialNB()
    clf.fit(X_vec, y_train)
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(clf, f)
    with open(VECTOR_FILE, "wb") as f:
        pickle.dump(vectorizer, f)
    print("✅ Đã huấn luyện AI xong!")
else:
    with open(MODEL_FILE, "rb") as f:
        clf = pickle.load(f)
    with open(VECTOR_FILE, "rb") as f:
        vectorizer = pickle.load(f)
    print("✅ Đã load AI có sẵn!")

# ------------------ TỪ CẤM ------------------
if os.path.exists(DATA_FILE):
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        BAD_WORDS = set(json.load(f))
else:
    BAD_WORDS = set()

def save_words():
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(list(BAD_WORDS), f, ensure_ascii=False)

def is_admin(user_id):
    return user_id in ADMIN_CHAT_IDS

# ------------------ LOG VI PHẠM ------------------
def add_violation(user_name, user_id, text, reason, score=0):
    violation = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user_name": user_name,
        "user_id": user_id,
        "text": text[:100],
        "reason": reason,
        "score": score
    }
    violations.insert(0, violation)
    if len(violations) > 100:
        violations.pop()
    save_logs()
    socketio.emit('new_violation', violation)

# ------------------ BOT TELEGRAM ------------------
async def add_word(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not is_admin(user_id):
        await update.message.reply_text("❌ Bạn không phải admin bot!")
        return
    if not context.args:
        await update.message.reply_text("⚠️ Cách dùng: `/addword từ_cấm`", parse_mode="Markdown")
        return
    word = " ".join(context.args).lower()
    BAD_WORDS.add(word)
    save_words()
    await update.message.reply_text(f"✅ Đã thêm từ cấm: `{word}`", parse_mode="Markdown")

async def del_word(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not is_admin(user_id):
        await update.message.reply_text("❌ Bạn không phải admin bot!")
        return
    if not context.args:
        await update.message.reply_text("⚠️ Cách dùng: `/delword từ_cấm`", parse_mode="Markdown")
        return
    word = " ".join(context.args).lower()
    if word in BAD_WORDS:
        BAD_WORDS.remove(word)
        save_words()
        await update.message.reply_text(f"✅ Đã xóa từ cấm: `{word}`", parse_mode="Markdown")
    else:
        await update.message.reply_text(f"❌ Không tìm thấy từ `{word}`", parse_mode="Markdown")

async def list_words(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if not is_admin(user_id):
        await update.message.reply_text("❌ Chỉ admin mới xem được!")
        return
    if not BAD_WORDS:
        await update.message.reply_text("📭 Chưa có từ cấm nào.")
        return
    word_list = "\n".join([f"• {w}" for w in sorted(BAD_WORDS)])
    await update.message.reply_text(f"🚫 **Từ cấm:**\n{word_list}", parse_mode="Markdown")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    await update.message.reply_text(
        "🤖 **Bot lọc tin nhắn nhóm**\n\n"
        "✅ Bot đang hoạt động!\n"
        f"👤 Bạn {'là admin' if is_admin(user_id) else 'không phải admin'}\n\n"
        "🔧 **Lệnh:**\n"
        "➜ `/addword từ` - Thêm từ cấm\n"
        "➜ `/delword từ` - Xóa từ cấm\n"
        "➜ `/badwords` - Xem danh sách\n\n"
        "🌐 **Dashboard:** http://localhost:5000",
        parse_mode="Markdown"
    )

async def filter_messages(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.effective_message
    user = message.from_user
    chat = message.chat
    text = message.text or message.caption or ""
    
    if not text or chat.type not in ["group", "supergroup"]:
        return
    if user.id in ADMIN_CHAT_IDS:
        return
    
    for bad_word in BAD_WORDS:
        if re.search(rf'\b{re.escape(bad_word)}\b', text, re.IGNORECASE):
            await message.delete()
            add_violation(user.full_name, user.id, text, f"Từ cấm: {bad_word}")
            await chat.send_message(f"⚠️ {user.mention_html()} dùng từ cấm: `{bad_word}`", parse_mode="HTML")
            return
    
    try:
        X_test = vectorizer.transform([text])
        toxic_score = clf.predict_proba(X_test)[0][1] if len(clf.predict_proba(X_test)[0]) > 1 else 0
        if toxic_score > 0.7:
            await message.delete()
            add_violation(user.full_name, user.id, text, "AI phát hiện nội dung xấu", toxic_score)
            await chat.send_message(
                f"🤖 AI phát hiện nội dung xấu ({toxic_score:.2f})\n⚠️ {user.mention_html()}",
                parse_mode="HTML"
            )
    except:
        pass

def run_bot():
    app_bot = Application.builder().token(TOKEN).build()
    app_bot.add_handler(CommandHandler("start", start))
    app_bot.add_handler(CommandHandler("addword", add_word))
    app_bot.add_handler(CommandHandler("delword", del_word))
    app_bot.add_handler(CommandHandler("badwords", list_words))
    app_bot.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, filter_messages))
    print("🤖 Bot Telegram đã chạy!")
    app_bot.run_polling()

# ------------------ DASHBOARD WEB ------------------
@app_flask.route('/')
def index():
    return render_template('dashboard.html', words=list(BAD_WORDS), violations=violations)

@app_flask.route('/api/words', methods=['GET'])
def get_words():
    return jsonify(list(BAD_WORDS))

@app_flask.route('/api/words', methods=['POST'])
def add_word_api():
    data = request.get_json()
    word = data.get('word', '').lower().strip()
    if word:
        BAD_WORDS.add(word)
        save_words()
        socketio.emit('word_added', word)
        return jsonify({'success': True, 'word': word})
    return jsonify({'success': False}), 400

@app_flask.route('/api/words/<word>', methods=['DELETE'])
def delete_word_api(word):
    word = word.lower()
    if word in BAD_WORDS:
        BAD_WORDS.remove(word)
        save_words()
        socketio.emit('word_deleted', word)
        return jsonify({'success': True})
    return jsonify({'success': False}), 404

@app_flask.route('/api/stats')
def get_stats():
    return jsonify({
        'total_words': len(BAD_WORDS),
        'total_violations': len(violations),
        'today_violations': len([v for v in violations if v['time'].startswith(datetime.now().strftime("%Y-%m-%d"))])
    })

# ------------------ CHẠY CẢ BOT VÀ DASHBOARD ------------------
if __name__ == "__main__":
    bot_thread = threading.Thread(target=run_bot)
    bot_thread.daemon = True
    bot_thread.start()
    
    os.makedirs("templates", exist_ok=True)
    
    print(f"🌐 Dashboard chạy tại: http://localhost:{DASHBOARD_PORT}")
    socketio.run(app_flask, host='0.0.0.0', port=DASHBOARD_PORT, debug=False, allow_unsafe_werkzeug=True)
