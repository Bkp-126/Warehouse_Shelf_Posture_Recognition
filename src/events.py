"""Deterministic video-time event state machine; no model or Qt dependency."""

from dataclasses import dataclass, field, asdict


@dataclass
class Event:
    session_id: str
    event_id: str
    track_id: int
    event_type: str
    roi: str
    start_time: float
    end_time: float
    effective_duration: float = 0.0
    alert_time: float | None = None
    end_reason: str = ""
    evidence_paths: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


class Channel:
    def __init__(self, key, settings, factory):
        self.key, self.settings, self.factory = key, settings, factory
        self.event = None
        self.start = None
        self.effective = 0.0
        self.last_t = None
        self.last_value = None
        self.last_positive = None
        self.gap_start = None
        self.gap_kind = None

    def reset(self):
        self.event = None
        self.start = None
        self.effective = 0.0
        self.last_positive = None
        self.gap_start = None
        self.gap_kind = None

    def finish(self, reason):
        ev = self.event
        if ev:
            ev.end_time = self.last_positive
            ev.effective_duration = self.effective
            ev.end_reason = reason
        self.reset()
        return ev

    def update(self, t, value):
        if self.last_t is not None and t < self.last_t:
            raise ValueError("视频时间必须单调递增")
        messages = []
        s = self.settings
        temporal = s.mode == "stable"
        enter = s.enter_seconds if temporal else 0
        exit_delay = s.exit_seconds if temporal else 0
        missing = s.missing_seconds if temporal else 0
        # If no samples arrived for a long interval, don't count the gap as posture.
        if self.last_t is not None and t - self.last_t > max(missing, 0.5) + 1e-8:
            ev = self.finish("timestamp_gap")
            if ev:
                messages.append(("ended", ev))
            self.last_value = None
        if self.gap_start is not None:
            limit = missing if self.gap_kind is None else exit_delay
            if t - self.gap_start >= limit - 1e-8:
                ev = self.finish("unobservable" if self.gap_kind is None else "recovered")
                if ev:
                    messages.append(("ended", ev))
        if value is True:
            if self.start is None:
                self.start = t
            if self.last_value is True and self.last_t is not None:
                self.effective += t - self.last_t
            self.last_positive = t
            self.gap_start = None
            self.gap_kind = None
            if self.event is None and self.effective >= enter - 1e-8:
                self.event = self.factory(self.key, self.start, t)
                messages.append(("started", self.event))
            if self.event:
                self.event.end_time = t
                self.event.effective_duration = self.effective
                if (
                    self.key[1] == "bend"
                    and self.event.alert_time is None
                    and self.effective >= s.alert_seconds - 1e-8
                ):
                    self.event.alert_time = t
                    messages.append(("alert", self.event))
        elif self.start is not None and self.event is None:
            # Confirmation needs a continuous observed candidate, not accumulated isolated pulses.
            self.reset()
        elif self.start is not None:
            if self.gap_start is None or self.gap_kind is not value:
                self.gap_start = t
                self.gap_kind = value
            limit = missing if value is None else exit_delay
            if limit == 0:
                ev = self.finish("unobservable" if value is None else "recovered")
                if ev:
                    messages.append(("ended", ev))
        self.last_t, self.last_value = t, value
        return messages


class EventMachine:
    def __init__(self, session_id, settings):
        self.session_id, self.settings = session_id, settings
        self.channels = {}
        self.events = []
        self.seq = 0

    def factory(self, key, start, t):
        self.seq += 1
        ev = Event(
            self.session_id, f"{self.session_id}_{self.seq:06d}", key[0], key[1], key[2], start, t
        )
        self.events.append(ev)
        return ev

    def active(self, key):
        ch = self.channels.get(key)
        return ch is not None and ch.event is not None

    def update(self, t, observations):
        messages = []
        for key in sorted(set(self.channels) | set(observations)):
            if key not in self.channels:
                self.channels[key] = Channel(key, self.settings, self.factory)
            ch = self.channels[key]
            changes = ch.update(t, observations.get(key))
            if key not in observations:
                for kind, event in changes:
                    if kind == "ended" and event.end_reason == "unobservable":
                        event.end_reason = "track_lost"
            messages.extend(changes)
            if key not in observations and ch.start is None:
                del self.channels[key]
        return messages

    def close(self, reason):
        messages = []
        for ch in self.channels.values():
            ev = ch.finish(reason)
            if ev:
                messages.append(("ended", ev))
        self.channels.clear()
        return messages
