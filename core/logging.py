import logging
import sys

class _HotwordFilter(logging.Filter):
    def filter(self, record):
        return ('Attempting to parse hotwords' not in record.getMessage() and
                'Hotword list:' not in record.getMessage() and 'rtf_avg:' not in record.getMessage())

def setup_logging(log_path: str):
    # 1. 清空旧 handler
    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)

    # 2. 创建两个真正的 handler
    file_hdl = logging.FileHandler(log_path, encoding='utf-8') # 写入文件中
    console_hdl = logging.StreamHandler(sys.stdout) # 输出到控制台

    # 3. 将过滤器挂到 handler上
    file_hdl.addFilter(_HotwordFilter())
    console_hdl.addFilter(_HotwordFilter())

    # 4. 配置
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[file_hdl, console_hdl]
    )

    # 日志降噪
    logging.getLogger("ai-voice-analysis-service").setLevel(logging.WARNING)
    logging.getLogger("python_multipart.multipart").setLevel(logging.WARNING)
    logging.getLogger("faster_whisper").setLevel(logging.WARNING)

