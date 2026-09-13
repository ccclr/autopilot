#!/bin/bash
set -e

IFACE=$(ip -4 -o addr show to 10.10.1.3 | awk '{print $2; exit}')
if [ -z "$IFACE" ]; then
  echo "Cannot find interface for 10.10.1.3" >&2
  exit 1
fi
echo "Using interface $IFACE for 10.10.1.3"

sudo tc qdisc del dev $IFACE root 2>/dev/null || true

sudo tc qdisc add dev $IFACE root handle 1: prio


sudo tc qdisc add dev $IFACE \
    parent 1:1 \
    handle 10: \
    netem delay 40ms


sudo tc filter add dev $IFACE \
    protocol ip \
    parent 1:0 \
    prio 1 \
    u32 match ip dst 10.10.1.1 \
    flowid 1:1


sudo tc qdisc add dev $IFACE \
    parent 1:2 \
    handle 20: \
    netem delay 40ms


sudo tc filter add dev $IFACE \
    protocol ip \
    parent 1:0 \
    prio 2 \
    u32 match ip dst 10.10.1.2 \
    flowid 1:2


sudo tc qdisc add dev $IFACE \
    parent 1:3 \
    handle 30: \
    netem delay 50ms


sudo tc filter add dev $IFACE \
    protocol ip \
    parent 1:0 \
    prio 3 \
    u32 match ip dst 10.10.1.4 \
    flowid 1:3

