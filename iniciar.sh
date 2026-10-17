#!/usr/bin/env bash
# Inicia el servidor de la página y muestra la dirección (y un QR) para abrirla en el celular.
cd "$(dirname "$0")"

PUERTO=8000
IP=$(ip -4 -o addr show scope global | awk '$2 !~ /^(docker|br-|veth|lo)/ {split($4, a, "/"); print a[1]; exit}')

if [ -z "$IP" ]; then
  echo "No estás conectado a ninguna red Wi-Fi."
  exit 1
fi

#Si quedó un servidor viejo encendido en el puerto (p. ej. se cerró la terminal sin Ctrl+C), apagarlo
VIEJO=$(pgrep -f "http.server $PUERTO")
if [ -n "$VIEJO" ]; then
  echo "Había un servidor anterior encendido, lo apago..."
  kill -CONT $VIEJO 2>/dev/null   #por si quedó pausado con Ctrl+Z
  kill $VIEJO 2>/dev/null
  sleep 1
  pgrep -f "http.server $PUERTO" >/dev/null && kill -9 $VIEJO 2>/dev/null
fi

URL="http://$IP:$PUERTO"
echo
echo "  En el celular (mismo Wi-Fi) abre:  $URL"
echo
command -v qrencode >/dev/null && qrencode -t ansiutf8 "$URL"
echo "  Para apagar el servidor: Ctrl + C"
echo

python3 -m http.server "$PUERTO" --bind 0.0.0.0
