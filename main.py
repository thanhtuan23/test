import os
import time
import numpy as np
from datetime import datetime
import scapy.all as scapy
from tensorflow.keras.models import load_model

# 🔹 Load the trained AI model
MODEL_PATH = "train_ai/best_model_v1.keras"  # Change the path if needed
model = load_model(MODEL_PATH)

# List to store blocked IPs to avoid blocking multiple times
blocked_ips = set()

# 📌 Extract features from the packet
def extract_features(packet):
    try:
        features = [
            packet[scapy.IP].len,   # Packet length
            packet[scapy.IP].ttl,   # Time to live
            packet[scapy.TCP].sport if packet.haslayer(scapy.TCP) else 0,  # Source port
            packet[scapy.TCP].dport if packet.haslayer(scapy.TCP) else 0   # Destination port
        ]

        # Simulate additional data to reach size 72 (replace with actual logic)
        while len(features) < 72:
            features.append(0)

        # Reshape the input data to match the model (None, 72, 1)
        features = np.array(features).reshape(1, 72, 1)
        return features

    except Exception as e:
        print(f"⚠️ Feature extraction error: {e}")
        return None

# 📌 Function to detect DDoS attacks
def detect_ddos(packet):
    features = extract_features(packet)
    if features is None:
        return

    prediction = model.predict(features)
    if prediction[0][0] > 0.8:  # Threshold to determine attack
        src_ip = packet[scapy.IP].src
        if src_ip not in blocked_ips:
            print(f"⚠️ Warning! DDoS detected from {src_ip}")
            block_ip(src_ip)
            log_blocked_ip(src_ip)
            blocked_ips.add(src_ip)

# 📌 Function to block IP (supports both Linux and Windows)
def block_ip(ip):
    if os.name == 'posix':  # Linux
        command = f"iptables -A INPUT -s {ip} -j DROP"
    elif os.name == 'nt':  # Windows
        command = f"netsh advfirewall firewall add rule name=\"Block IP {ip}\" dir=in action=block remoteip={ip}"
    else:
        print(f"⚠️ Unsupported operating system: {os.name}")
        return

    os.system(command)
    print(f"🚫 IP blocked: {ip}")

# 📌 Log blocked IPs
def log_blocked_ip(ip):
    with open('blocked_ips.log', 'a') as log_file:
        log_file.write(f"{datetime.now()} - Blocked IP: {ip}\n")

# 📌 Run the network monitoring system
def main():
    print("🔍 Monitoring network for DDoS attacks...")
    scapy.sniff(prn=detect_ddos, store=False)

if __name__ == "__main__":
    main()
