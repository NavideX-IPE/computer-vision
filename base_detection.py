"""
NavideX-IPE - Computer Vision base station

Receives the drone camera stream, runs the pending YOLO model, and when a
configured target is detected:
  1. reads and saves the latest GPS position;
  2. saves the triggering image;
  3. sends RTL (Return-to-Launch) to the drone.

The YOLO model is intentionally expected to be `modelo.pt` for now.
"""

from pathlib import Path
from datetime import datetime, timezone
import sys

# ========================= USER CONFIGURATION =========================

# Edit this to the Wi-Fi stream port used by the drone.
WIFI_STREAM_PORT = 5000

# Connection used by MAVLink. Edit for the actual base/SITL setup.
# Examples: "udp:127.0.0.1:14550", "tcp:127.0.0.1:5760"
MAVLINK_CONNECTION = "udp:127.0.0.1:14550"

MODEL_PATH = Path("modelo.pt")
CONFIDENCE_THRESHOLD = 0.50

SAVE_DIR = Path("detections")
IMAGE_FILENAME = "detection.jpg"
GPS_FILENAME = "last_gps.txt"

# Set to the YOLO class IDs that represent the target. Leave None to accept
# any detected class from the currently pending model.
TARGET_CLASS_IDS = None

# ======================================================================


def load_yolo_model():
    """Load modelo.pt as a YOLO model and fail clearly if it cannot be loaded."""
    try:
        from ultralytics import YOLO
    except Exception as exc:
        print(f"[ERRO] Deu merda importando Ultralytics/YOLO: {exc}")
        sys.exit(1)

    try:
        model = YOLO(str(MODEL_PATH))
        print(f"[OK] Modelo YOLO carregado: {MODEL_PATH}")
        return model
    except Exception as exc:
        print(f"[ERRO] Deu merda ao carregar {MODEL_PATH} como YOLO: {exc}")
        sys.exit(1)


def connect_mavlink():
    """Connect to the drone/autopilot through MAVLink."""
    try:
        from pymavlink import mavutil
    except Exception as exc:
        print(f"[ERRO] Deu merda importando pymavlink: {exc}")
        sys.exit(1)

    try:
        connection = mavutil.mavlink_connection(MAVLINK_CONNECTION)
        print(f"[INFO] Conectando ao MAVLink em {MAVLINK_CONNECTION}...")
        connection.wait_heartbeat(timeout=10)
        print("[OK] Heartbeat MAVLink recebido.")
        return connection
    except Exception as exc:
        print(f"[ERRO] Nao foi possivel conectar ao MAVLink: {exc}")
        sys.exit(1)


def get_latest_gps(mav):
    """Request/read the most recent GPS position available from MAVLink."""
    msg = mav.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=2)
    if msg is None:
        return None

    return {
        "latitude": msg.lat / 1e7,
        "longitude": msg.lon / 1e7,
        "altitude_m": msg.alt / 1000.0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def save_detection(frame, gps):
    """Save the detected image and its GPS coordinates."""
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    image_path = SAVE_DIR / IMAGE_FILENAME
    gps_path = SAVE_DIR / GPS_FILENAME

    if not frame is None:
        import cv2
        if not cv2.imwrite(str(image_path), frame):
            raise RuntimeError(f"Falha ao salvar imagem em {image_path}")

    if gps is not None:
        gps_path.write_text(
            "latitude={latitude}\n"
            "longitude={longitude}\n"
            "altitude_m={altitude_m}\n"
            "timestamp={timestamp}\n".format(**gps),
            encoding="utf-8",
        )
    else:
        gps_path.write_text(
            "GPS indisponivel no momento da deteccao.\n", encoding="utf-8")

    print(f"[OK] Imagem salva em: {image_path}")
    print(f"[OK] GPS salvo em: {gps_path}")


def send_rtl(mav):
    """Command the autopilot to Return-to-Launch."""
    mav.mav.command_long_send(
        mav.target_system,
        mav.target_component,
        mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH,
        0,
        0, 0, 0, 0, 0, 0, 0,
    )
    print("[OK] Comando RTL enviado ao drone.")


def detection_is_target(result):
    """Return True when the YOLO result contains a configured target."""
    if result.boxes is None or len(result.boxes) == 0:
        return False

    for box in result.boxes:
        confidence = float(box.conf[0])
        if confidence < CONFIDENCE_THRESHOLD:
            continue

        class_id = int(box.cls[0])
        if TARGET_CLASS_IDS is None or class_id in TARGET_CLASS_IDS:
            print(
                f"[ALVO] Classe={class_id}, "
                f"confianca={confidence:.3f}"
            )
            return True

    return False


def open_stream():
    """Open the drone Wi-Fi video stream."""
    import cv2

    # Placeholder URL: adjust to the drone's actual stream protocol/path.
    stream_url = f"http://127.0.0.1:{WIFI_STREAM_PORT}/stream"
    print(f"[INFO] Abrindo stream em {stream_url}")

    capture = cv2.VideoCapture(stream_url)
    if not capture.isOpened():
        raise RuntimeError(
            f"Nao foi possivel abrir a stream na porta {WIFI_STREAM_PORT}. "
            "Edite WIFI_STREAM_PORT e/ou a URL da stream conforme a implementacao do drone."
        )

    return capture


def main():
    model = load_yolo_model()
    mav = connect_mavlink()
    capture = open_stream()

    print("[OK] Base pronta. Monitorando stream...")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print(
                    "[AVISO] Falha ao receber frame da stream; tentando novamente...")
                continue

            results = model(frame, verbose=False)

            if not any(detection_is_target(result) for result in results):
                continue

            print("[ALVO] Objeto identificado. Interrompendo monitoramento.")

            # Capture GPS as close as possible to the detection event.
            gps = get_latest_gps(mav)
            if gps:
                print(
                    f"[GPS] lat={gps['latitude']:.7f}, "
                    f"lon={gps['longitude']:.7f}, "
                    f"alt={gps['altitude_m']:.2f} m"
                )
            else:
                print("[AVISO] Nao foi possivel obter GPS no momento da deteccao.")

            save_detection(frame, gps)
            send_rtl(mav)

            # One detection is enough to trigger RTL. Do not keep sending RTL.
            break

    except KeyboardInterrupt:
        print("\n[INFO] Encerrado pelo usuario.")
    finally:
        capture.release()
        mav.close()


if __name__ == "__main__":
    main()
