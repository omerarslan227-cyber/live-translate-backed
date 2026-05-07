from __future__ import annotations

import asyncio
import contextlib
import secrets
from dataclasses import dataclass, field
from typing import Any

from fastapi import WebSocket


@dataclass
class Client:
    websocket: WebSocket
    client_id: str
    room: str | None = None
    is_owner: bool = False


@dataclass
class Room:
    name: str
    private_code: str
    capacity: int
    owner_id: str
    members: dict[str, Client] = field(default_factory=dict)


class RoomManager:
    def __init__(self) -> None:
        self._rooms: dict[str, Room] = {}
        self._clients: dict[str, Client] = {}
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> Client:
        await websocket.accept()
        client = Client(websocket=websocket, client_id=secrets.token_urlsafe(12))
        async with self._lock:
            self._clients[client.client_id] = client
        await self.send(client, {"type": "welcome", "clientId": client.client_id})
        return client

    async def disconnect(self, client: Client) -> None:
        room_name: str | None = None
        room_closed = False
        notify_clients: list[Client] = []
        async with self._lock:
            self._clients.pop(client.client_id, None)
            if client.room and client.room in self._rooms:
                room = self._rooms[client.room]
                room.members.pop(client.client_id, None)
                room_name = room.name
                room_closed = client.is_owner or not room.members
                notify_clients = [
                    member
                    for member in room.members.values()
                    if room_closed or member.client_id != client.client_id
                ]
                if room_closed:
                    self._rooms.pop(room.name, None)
                client.room = None

        if room_name:
            payload = {"type": "room_closed" if room_closed else "member_left", "room": room_name}
            for target in notify_clients:
                await self.send(target, payload)

    async def handle(self, client: Client, message: dict[str, Any]) -> None:
        msg_type = str(message.get("type") or "")
        if msg_type == "create_room":
            await self._create_room(client, message)
            return
        if msg_type == "request_join":
            await self._request_join(client, message)
            return
        if msg_type == "join_decision":
            await self._join_decision(client, message)
            return
        if msg_type == "leave_room":
            await self.disconnect(client)
            await self.send(client, {"type": "left_room"})
            return
        if msg_type == "chat_message":
            await self._chat_message(client, message)
            return

        room = str(message.get("room") or client.room or "")
        if not room:
            await self.send_error(client, "Room is required")
            return
        payload = dict(message)
        payload.setdefault("room", room)
        payload.setdefault("senderId", client.client_id)
        await self.broadcast(room, payload, exclude_client_id=client.client_id)

    async def _create_room(self, client: Client, message: dict[str, Any]) -> None:
        room_name = str(message.get("room") or "").strip()
        private_code = str(message.get("privateCode") or "").strip()
        capacity = int(message.get("capacity") or 2)
        if not room_name or not private_code:
            await self.send_error(client, "Room and privateCode are required")
            return
        async with self._lock:
            old_room = self._rooms.get(room_name)
            if old_room and old_room.members:
                await self.send_error(client, "Room already exists")
                return
            room = Room(room_name, private_code, max(2, capacity), client.client_id)
            room.members[client.client_id] = client
            client.room = room_name
            client.is_owner = True
            self._rooms[room_name] = room
            member_count = len(room.members)
        await self.send(client, {"type": "room_created", "room": room_name, "memberCount": member_count})

    async def _request_join(self, client: Client, message: dict[str, Any]) -> None:
        room_name = str(message.get("room") or "").strip()
        private_code = str(message.get("privateCode") or "").strip()
        async with self._lock:
            room = self._rooms.get(room_name)
            if not room:
                await self.send_error(client, "Room not found")
                return
            if room.private_code != private_code:
                await self.send_error(client, "Invalid private code")
                return
            if len(room.members) >= room.capacity:
                await self.send_error(client, "Room is full")
                return
            owner = room.members.get(room.owner_id)
        if owner is None:
            await self.send_error(client, "Room owner is offline")
            return
        await self.send(
            owner,
            {
                "type": "join_request",
                "room": room_name,
                "requesterId": client.client_id,
            },
        )

    async def _join_decision(self, client: Client, message: dict[str, Any]) -> None:
        if not client.is_owner:
            await self.send_error(client, "Only owner can accept joins")
            return
        room_name = str(message.get("room") or client.room or "")
        requester_id = str(message.get("requesterId") or "")
        accept = message.get("accept") is not False
        async with self._lock:
            room = self._rooms.get(room_name)
            requester = self._clients.get(requester_id)
            if not room or not requester:
                return
            if not accept:
                target = requester
                joined = False
                member_count = len(room.members)
            elif len(room.members) >= room.capacity:
                target = requester
                joined = False
                member_count = len(room.members)
            else:
                requester.room = room_name
                requester.is_owner = False
                room.members[requester.client_id] = requester
                target = requester
                joined = True
                member_count = len(room.members)
        if not accept or not joined:
            await self.send(target, {"type": "join_rejected", "room": room_name})
            return
        await self.send(target, {"type": "join_accepted", "room": room_name, "memberCount": member_count})
        await self.broadcast(
            room_name,
            {"type": "member_joined", "room": room_name, "memberCount": member_count, "clientId": requester_id},
            exclude_client_id=requester_id,
        )
        await self.broadcast(
            room_name,
            {"type": "room_state", "room": room_name, "memberCount": member_count},
            exclude_client_id=None,
        )

    async def _chat_message(self, client: Client, message: dict[str, Any]) -> None:
        room = str(message.get("room") or client.room or "")
        text = str(message.get("text") or "").strip()
        if not room or not text:
            return
        payload = {
            "type": "chat_message",
            "room": room,
            "senderId": client.client_id,
            "text": text,
            "translatedText": str(message.get("translatedText") or ""),
        }
        await self.broadcast(room, payload, exclude_client_id=None)

    async def broadcast(
        self,
        room_name: str,
        payload: dict[str, Any],
        *,
        exclude_client_id: str | None,
    ) -> None:
        async with self._lock:
            room = self._rooms.get(room_name)
            clients = list(room.members.values()) if room else []
        for target in clients:
            if exclude_client_id and target.client_id == exclude_client_id:
                continue
            await self.send(target, payload)

    async def send(self, client: Client, payload: dict[str, Any]) -> None:
        with contextlib.suppress(Exception):
            await client.websocket.send_json(payload)

    async def send_error(self, client: Client, message: str) -> None:
        await self.send(client, {"type": "error", "message": message})
