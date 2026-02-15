#!/usr/bin/env python3
"""Generate words.json and bigrams.json corpus files.

This script builds the bundled corpus from public word frequency data.
It creates a curated set of ~100k English words with Zipf frequencies
and ~50k common bigrams with log-probabilities.

Usage:
    python scripts/build_corpus.py

The output files are written to src/txtpand/corpus/.
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

# Common English words with approximate Zipf frequencies (log10 of frequency per billion words)
# This is a seed set; a full build would pull from a public corpus like Google Ngrams or SUBTLEX.
# Zipf scale: 7 = extremely common ("the"), 1 = very rare

SEED_WORDS: dict[str, float] = {
    # Articles/determiners
    "the": 7.0, "a": 6.8, "an": 6.0, "this": 6.3, "that": 6.5,
    "these": 5.5, "those": 5.3, "my": 6.0, "your": 5.8, "his": 5.7,
    "her": 5.6, "its": 5.2, "our": 5.4, "their": 5.5, "some": 5.6,
    "any": 5.4, "no": 5.8, "every": 5.1, "each": 5.0, "all": 6.0,
    "both": 5.2, "few": 5.0, "several": 4.5, "many": 5.5, "much": 5.4,

    # Pronouns
    "i": 6.8, "me": 6.3, "we": 6.2, "us": 5.5, "you": 6.7,
    "he": 6.3, "him": 5.7, "she": 6.0, "they": 6.2, "them": 5.7,
    "it": 6.7, "what": 6.3, "which": 5.8, "who": 5.8, "whom": 4.0,
    "where": 5.8, "when": 5.9, "why": 5.5, "how": 5.9,
    "myself": 4.8, "yourself": 4.5, "himself": 4.7, "herself": 4.5,
    "itself": 4.5, "ourselves": 4.0, "themselves": 4.3,

    # Prepositions/conjunctions
    "in": 6.8, "on": 6.5, "at": 6.3, "to": 7.0, "for": 6.7,
    "with": 6.5, "from": 6.3, "by": 6.2, "of": 7.0, "about": 5.8,
    "into": 5.5, "through": 5.3, "during": 4.8, "before": 5.3,
    "after": 5.3, "above": 4.5, "below": 4.3, "between": 5.0,
    "under": 4.8, "over": 5.2, "out": 5.8, "up": 5.8, "down": 5.5,
    "off": 5.2, "and": 7.0, "but": 6.3, "or": 6.2, "nor": 3.8,
    "so": 6.0, "yet": 5.0, "if": 6.2, "then": 5.5, "than": 5.5,
    "because": 5.3, "since": 4.8, "while": 5.0, "although": 4.3,
    "though": 4.8, "unless": 4.2, "until": 4.8, "as": 6.3,
    "also": 5.5, "too": 5.3, "very": 5.5, "quite": 4.5,
    "rather": 4.3, "even": 5.3, "just": 5.8, "only": 5.5,
    "still": 5.2, "already": 4.8, "almost": 4.8,

    # Common verbs
    "be": 6.8, "is": 6.8, "am": 5.5, "are": 6.5, "was": 6.5,
    "were": 5.8, "been": 5.5, "being": 5.0, "have": 6.5, "has": 6.2,
    "had": 6.0, "having": 4.5, "do": 6.3, "does": 5.5, "did": 5.8,
    "doing": 4.8, "done": 5.0, "will": 6.2, "would": 6.0, "shall": 4.0,
    "should": 5.5, "may": 5.3, "might": 5.0, "can": 6.0, "could": 5.8,
    "must": 5.0, "need": 5.5, "want": 5.5, "like": 5.8, "know": 5.8,
    "think": 5.5, "thought": 5.2, "say": 5.8, "said": 5.8, "tell": 5.3,
    "told": 5.0, "give": 5.3, "gave": 4.8, "given": 4.8, "take": 5.5,
    "took": 5.0, "taken": 4.5, "make": 5.8, "made": 5.5, "go": 5.8,
    "went": 5.2, "gone": 4.8, "going": 5.5, "come": 5.5, "came": 5.0,
    "get": 5.8, "got": 5.5, "see": 5.5, "saw": 4.8, "seen": 4.8,
    "look": 5.3, "find": 5.3, "found": 5.2, "put": 5.3, "keep": 5.2,
    "kept": 4.5, "let": 5.3, "begin": 4.5, "began": 4.5, "begun": 3.8,
    "show": 5.0, "shown": 4.3, "try": 5.2, "tried": 4.8, "turn": 4.8,
    "turned": 4.5, "use": 5.5, "used": 5.3, "using": 5.0, "run": 5.0,
    "ran": 4.3, "work": 5.5, "worked": 4.8, "working": 5.0,
    "call": 5.0, "called": 5.0, "ask": 5.0, "asked": 4.8,
    "move": 4.8, "moved": 4.5, "live": 5.0, "lived": 4.3,
    "help": 5.2, "helped": 4.3, "start": 5.0, "started": 4.8,
    "stop": 4.8, "stopped": 4.3, "open": 4.8, "opened": 4.3,
    "close": 4.5, "closed": 4.3, "play": 4.8, "played": 4.3,
    "read": 5.0, "write": 5.0, "wrote": 4.5, "written": 4.5,
    "learn": 4.8, "learned": 4.3, "change": 5.0, "changed": 4.5,
    "follow": 4.5, "followed": 4.0, "create": 4.8, "created": 4.5,
    "speak": 4.5, "spoke": 4.0, "stand": 4.8, "stood": 4.3,
    "hear": 4.8, "heard": 4.8, "bring": 4.8, "brought": 4.5,
    "hold": 4.8, "held": 4.5, "set": 5.0, "sit": 4.5, "sat": 4.3,
    "pay": 4.8, "paid": 4.5, "meet": 4.5, "met": 4.3,
    "send": 4.8, "sent": 4.5, "build": 4.8, "built": 4.5,
    "stay": 4.5, "fall": 4.5, "fell": 4.0, "cut": 4.5,
    "reach": 4.3, "kill": 4.3, "remain": 4.0, "suggest": 4.0,
    "raise": 4.0, "pass": 4.5, "sell": 4.3, "require": 4.3,
    "report": 4.5, "decide": 4.3, "pull": 4.3, "develop": 4.5,
    "happen": 4.5, "believe": 4.8, "feel": 5.0, "felt": 4.5,
    "become": 5.0, "became": 4.5, "leave": 5.0, "left": 5.0,
    "mean": 5.0, "meant": 4.5, "understand": 4.8, "watch": 4.5,
    "seem": 4.8, "add": 4.5, "remember": 4.5, "love": 5.0,
    "consider": 4.3, "appear": 4.3, "buy": 4.5, "wait": 4.5,
    "serve": 4.0, "die": 4.3, "expect": 4.3, "win": 4.5,
    "won": 4.3, "lose": 4.3, "lost": 4.8, "handle": 4.0,
    "implement": 4.0, "fix": 4.3, "explain": 4.3, "generate": 4.0,

    # Common nouns
    "time": 6.0, "year": 5.5, "people": 5.8, "way": 5.8,
    "day": 5.5, "man": 5.5, "woman": 5.0, "child": 5.0,
    "children": 4.8, "world": 5.5, "life": 5.5, "hand": 5.2,
    "part": 5.2, "place": 5.2, "case": 5.0, "week": 4.8,
    "company": 4.8, "system": 5.0, "program": 4.8, "question": 5.0,
    "work": 5.5, "government": 4.8, "number": 5.0, "night": 5.0,
    "point": 5.0, "home": 5.2, "water": 5.0, "room": 5.0,
    "mother": 5.0, "area": 4.8, "money": 5.0, "story": 4.8,
    "fact": 5.0, "month": 4.8, "lot": 5.0, "right": 5.5,
    "study": 4.5, "book": 5.0, "eye": 4.8, "job": 4.8,
    "word": 5.0, "business": 4.8, "issue": 4.5, "side": 4.8,
    "kind": 5.0, "head": 5.0, "house": 5.0, "service": 4.5,
    "friend": 4.8, "father": 4.8, "power": 4.8, "hour": 4.8,
    "game": 4.8, "line": 5.0, "end": 5.2, "member": 4.3,
    "law": 4.5, "car": 4.8, "city": 4.8, "community": 4.3,
    "name": 5.2, "president": 4.3, "team": 4.5, "minute": 4.8,
    "idea": 4.8, "body": 4.8, "information": 4.8, "back": 5.5,
    "parent": 4.3, "face": 5.0, "others": 4.8, "level": 4.5,
    "office": 4.5, "door": 4.8, "health": 4.5, "person": 4.8,
    "art": 4.3, "war": 4.5, "history": 4.3, "party": 4.5,
    "result": 4.5, "morning": 4.8, "reason": 4.8, "research": 4.5,
    "girl": 4.8, "guy": 4.8, "moment": 4.8, "air": 4.5,
    "teacher": 4.3, "force": 4.3, "education": 4.3, "thing": 5.5,
    "things": 5.5, "something": 5.5, "nothing": 5.2, "everything": 5.0,
    "anything": 5.0, "someone": 4.8, "everyone": 4.5, "example": 4.5,

    # Common adjectives
    "good": 5.8, "new": 5.5, "first": 5.5, "last": 5.2,
    "long": 5.2, "great": 5.3, "little": 5.3, "own": 5.2,
    "other": 5.8, "old": 5.2, "right": 5.5, "big": 5.0,
    "high": 5.0, "different": 4.8, "small": 5.0, "large": 4.8,
    "next": 5.0, "early": 4.8, "young": 4.8, "important": 4.8,
    "few": 5.0, "public": 4.5, "bad": 5.0, "same": 5.0,
    "able": 4.8, "sure": 5.0, "real": 4.8, "best": 5.0,
    "better": 5.0, "true": 4.8, "hard": 4.8, "possible": 4.5,
    "full": 4.8, "free": 4.8, "strong": 4.5, "wrong": 4.8,
    "happy": 4.5, "sorry": 4.8, "simple": 4.3, "clear": 4.5,
    "easy": 4.5, "specific": 4.0, "whole": 4.8, "current": 4.3,
    "nice": 4.5, "certain": 4.3, "personal": 4.0, "open": 4.8,
    "available": 4.3, "ready": 4.5, "human": 4.3, "local": 4.0,
    "short": 4.5, "single": 4.3, "common": 4.3, "white": 4.5,
    "black": 4.5, "red": 4.5, "blue": 4.3, "green": 4.3,
    "quick": 4.5, "slow": 4.0, "fast": 4.5, "brown": 4.0,
    "bright": 4.0, "dark": 4.3, "deep": 4.3, "wide": 4.0,
    "thin": 4.3, "thick": 3.8, "flat": 3.8, "round": 3.8,
    "quiet": 4.0, "loud": 3.8, "soft": 3.8, "rough": 3.5,
    "smooth": 3.5, "warm": 4.0, "cool": 4.0, "cold": 4.3,
    "hot": 4.3, "dry": 3.8, "wet": 3.5, "fresh": 3.8,
    "clean": 4.0, "dirty": 3.5, "empty": 4.0, "rich": 3.8,
    "poor": 3.8, "heavy": 4.0, "light": 4.5, "sharp": 3.5,
    "sweet": 3.8, "wild": 3.8, "tiny": 4.0, "huge": 4.3,
    "lazy": 3.5, "busy": 4.0, "ugly": 3.5, "pretty": 4.3,
    "lovely": 3.8, "silly": 3.5, "crazy": 4.0, "scary": 3.5,
    "funny": 4.0, "angry": 4.0, "sad": 4.0, "sick": 3.8,
    "safe": 4.0, "dangerous": 3.8, "strange": 3.8, "weird": 4.0,
    "normal": 4.0, "regular": 3.8, "proper": 3.8, "fair": 3.8,
    "rare": 3.8, "obvious": 3.8, "basic": 4.0, "complex": 4.0,
    "private": 3.5, "extra": 4.0, "main": 4.3, "major": 4.0,
    "minor": 3.5, "final": 4.3, "initial": 3.5, "primary": 3.5,
    "secondary": 3.0, "direct": 3.8, "legal": 3.8, "physical": 3.8,
    "mental": 3.5, "social": 4.0, "political": 3.8, "medical": 3.5,
    "natural": 4.0, "original": 3.8, "complete": 4.3, "entire": 3.8,
    "various": 3.8, "recent": 4.0, "serious": 4.0, "popular": 3.8,
    "successful": 3.5, "significant": 3.5, "similar": 4.0,
    "traditional": 3.3, "professional": 3.5, "international": 3.5,

    # Animals and nature
    "fox": 3.5, "wolf": 3.5, "bear": 3.8, "deer": 3.3,
    "horse": 3.8, "cow": 3.5, "pig": 3.3, "sheep": 3.3,
    "chicken": 3.5, "duck": 3.3, "rabbit": 3.3, "mouse": 3.5,
    "snake": 3.3, "lion": 3.3, "tiger": 3.3, "elephant": 3.0,
    "monkey": 3.3, "whale": 3.0, "shark": 3.0, "frog": 3.0,
    "bug": 3.8, "ant": 3.0, "bee": 3.0, "fly": 4.0,
    "grass": 3.5, "leaf": 3.3, "rock": 3.8, "sand": 3.3,
    "snow": 3.5, "rain": 3.8, "wind": 3.8, "storm": 3.5,
    "cloud": 3.8, "sky": 3.8, "earth": 3.8, "sea": 3.8,
    "lake": 3.5, "hill": 3.5, "valley": 3.3, "wave": 3.5,

    # Common adverbs
    "not": 6.5, "also": 5.5, "very": 5.5, "often": 4.5,
    "however": 4.5, "never": 5.0, "always": 5.0, "sometimes": 4.5,
    "now": 5.8, "here": 5.5, "there": 5.8, "today": 4.8,
    "again": 5.0, "once": 4.8, "well": 5.5, "really": 5.2,
    "actually": 4.8, "probably": 4.8, "maybe": 4.8, "please": 5.0,
    "yes": 5.5, "no": 5.8, "okay": 5.0, "ok": 5.0, "yeah": 5.0,

    # Tech/programming terms
    "code": 4.8, "file": 4.8, "data": 4.8, "function": 4.5,
    "class": 4.5, "method": 4.3, "variable": 4.0, "string": 4.3,
    "type": 4.8, "list": 4.5, "array": 4.0, "object": 4.5,
    "error": 4.5, "test": 4.5, "debug": 3.8, "server": 4.3,
    "client": 4.3, "database": 4.0, "table": 4.3, "query": 4.0,
    "api": 4.3, "url": 4.0, "web": 4.3, "app": 4.5,
    "application": 4.3, "software": 4.0, "hardware": 3.8,
    "computer": 4.3, "network": 4.0, "user": 4.5, "password": 3.8,
    "login": 3.8, "email": 4.3, "message": 4.5, "search": 4.5,
    "page": 4.5, "link": 4.0, "image": 4.3, "video": 4.3,
    "button": 4.0, "click": 4.0, "screen": 4.3, "window": 4.3,
    "browser": 3.8, "download": 3.8, "upload": 3.5, "install": 3.8,
    "update": 4.3, "version": 4.0, "feature": 4.3, "bug": 3.8,
    "config": 3.5, "configuration": 3.5, "setting": 4.0, "settings": 4.0,
    "project": 4.5, "deploy": 3.5, "deployment": 3.5, "production": 3.8,
    "docker": 3.3, "container": 3.8, "kubernetes": 3.0, "cloud": 3.8,
    "package": 4.0, "module": 3.8, "library": 4.0, "framework": 3.8,
    "python": 3.8, "javascript": 3.5, "react": 3.5, "node": 3.8,
    "git": 3.5, "commit": 3.5, "branch": 3.8, "merge": 3.5,
    "repository": 3.3, "repo": 3.5, "pull": 4.3, "push": 3.8,
    "interface": 3.8, "component": 3.8, "template": 3.5,
    "prompt": 3.8, "model": 4.3, "token": 3.8, "response": 4.3,
    "request": 4.3, "input": 4.0, "output": 4.0, "process": 4.3,
    "thread": 3.5, "async": 3.0, "await": 3.0, "promise": 3.3,
    "callback": 3.0, "event": 4.0, "handler": 3.3, "listener": 3.0,
    "import": 3.8, "export": 3.5, "return": 4.0, "value": 4.5,
    "key": 4.5, "index": 4.0, "item": 4.0, "element": 3.8,
    "parameter": 3.5, "argument": 3.8, "option": 4.0, "flag": 3.5,
    "command": 4.0, "script": 3.8, "shell": 3.5, "terminal": 3.5,
    "directory": 3.5, "folder": 3.5, "path": 4.0, "root": 3.8,
    "permission": 3.5, "access": 4.0, "security": 3.8, "authentication": 3.3,
    "authorization": 3.0, "encryption": 3.0, "certificate": 3.3,
    "log": 3.8, "logging": 3.3, "monitor": 3.3, "metric": 3.0,
    "performance": 3.8, "memory": 4.0, "cache": 3.5, "storage": 3.5,
    "load": 3.8, "save": 4.0, "delete": 3.8, "remove": 4.0,
    "insert": 3.5, "select": 3.8, "filter": 3.8, "sort": 3.5,
    "count": 3.8, "total": 4.0, "average": 3.8, "maximum": 3.5,
    "minimum": 3.5, "size": 4.0, "length": 3.8, "width": 3.5,
    "height": 3.5, "format": 4.0, "parse": 3.3, "validate": 3.3,
    "check": 4.5, "verify": 3.5, "confirm": 3.8, "approve": 3.3,
    "review": 4.0, "submit": 3.8, "cancel": 3.5, "retry": 3.0,
    "timeout": 3.0, "status": 4.0, "state": 4.0, "context": 3.8,
    "scope": 3.5, "global": 3.5, "local": 4.0, "private": 3.5,
    "public": 4.5, "static": 3.3, "dynamic": 3.5, "abstract": 3.3,
    "virtual": 3.3, "override": 3.0, "extend": 3.3, "inherit": 3.0,
    "implement": 4.0, "define": 3.8, "declare": 3.3, "initialize": 3.0,
    "execute": 3.5, "compile": 3.3, "runtime": 3.3, "syntax": 3.3,
    "null": 3.5, "boolean": 3.0, "integer": 3.3, "float": 3.3,
    "char": 3.0, "byte": 3.0, "bit": 3.5, "binary": 3.3,
    "hex": 2.8, "json": 3.5, "xml": 3.0, "html": 3.5,
    "css": 3.3, "sql": 3.3, "regex": 3.0, "pattern": 3.8,
    "match": 4.0, "replace": 3.8, "split": 3.5, "join": 3.5,
    "map": 3.8, "reduce": 3.5, "loop": 3.5, "iterate": 3.0,
    "recursive": 2.8, "algorithm": 3.5, "structure": 4.0,
    "tree": 3.5, "graph": 3.3, "stack": 3.5, "queue": 3.3,
    "hash": 3.3, "encode": 3.0, "decode": 3.0, "compress": 3.0,
    "extract": 3.5, "transform": 3.3, "convert": 3.5, "migrate": 3.0,
    "backup": 3.3, "restore": 3.3, "sync": 3.3, "refresh": 3.3,
    "render": 3.3, "display": 3.8, "print": 3.8, "output": 4.0,
    "input": 4.0, "stream": 3.5, "buffer": 3.3, "channel": 3.5,
    "socket": 3.0, "port": 3.5, "host": 3.5, "domain": 3.5,
    "address": 4.0, "protocol": 3.3, "header": 3.5, "body": 4.8,
    "payload": 3.0, "token": 3.8, "session": 3.5, "cookie": 3.3,
    "middleware": 3.0, "proxy": 3.3, "gateway": 3.0, "router": 3.3,
    "endpoint": 3.3, "route": 3.5, "controller": 3.3, "service": 4.5,
    "layer": 3.5, "tier": 3.0, "stack": 3.5, "pipeline": 3.3,
    "workflow": 3.3, "task": 4.0, "job": 4.8, "worker": 3.5,
    "scheduler": 3.0, "trigger": 3.3, "hook": 3.3, "plugin": 3.3,
    "extension": 3.5, "addon": 2.8, "widget": 3.0, "dashboard": 3.3,
    "panel": 3.5, "tab": 3.5, "menu": 3.5, "toolbar": 3.0,
    "icon": 3.5, "badge": 3.0, "notification": 3.3, "alert": 3.5,
    "warning": 3.8, "error": 4.5, "success": 3.8, "failure": 3.5,
    "exception": 3.3, "crash": 3.3, "issue": 4.5, "ticket": 3.3,
    "priority": 3.3, "severity": 2.8, "critical": 3.3,
    "terraform": 3.0, "nginx": 3.0, "redis": 3.0, "postgres": 2.8,
    "mysql": 2.8, "mongodb": 2.8, "elasticsearch": 2.5,
    "linux": 3.3, "ubuntu": 2.8, "macos": 2.8, "windows": 3.8,
    "android": 3.3, "ios": 3.0, "mobile": 3.8, "desktop": 3.5,
    "laptop": 3.3, "tablet": 3.0, "device": 3.8, "sensor": 3.0,
    "bluetooth": 2.8, "wifi": 3.0, "internet": 4.0,

    # Conversational/AI interaction
    "please": 5.0, "thanks": 5.0, "thank": 5.0, "sorry": 4.8,
    "hello": 4.5, "hey": 4.5, "hi": 4.8, "bye": 3.8,
    "great": 5.3, "awesome": 4.3, "amazing": 4.3, "perfect": 4.3,
    "wonderful": 4.0, "terrible": 3.8, "horrible": 3.5,
    "interesting": 4.3, "exciting": 3.8, "boring": 3.5,
    "helpful": 4.0, "useful": 4.0, "important": 4.8,
    "necessary": 4.0, "essential": 3.8, "optional": 3.5,
    "recommend": 3.8, "suggest": 4.0, "prefer": 3.8,
    "instead": 4.5, "otherwise": 4.0, "regardless": 3.5,
    "however": 4.5, "therefore": 4.0, "furthermore": 3.5,
    "meanwhile": 3.5, "nevertheless": 3.3, "basically": 4.0,
    "essentially": 3.5, "generally": 3.8, "specifically": 3.8,
    "approximately": 3.5, "exactly": 4.3, "definitely": 4.3,

    # Numbers/ordinals
    "one": 5.8, "two": 5.5, "three": 5.2, "four": 5.0,
    "five": 4.8, "six": 4.5, "seven": 4.3, "eight": 4.3,
    "nine": 4.3, "ten": 4.5, "hundred": 4.3, "thousand": 4.0,
    "million": 4.0, "billion": 3.5, "second": 4.8, "third": 4.3,

    # Additional high-value expansion targets
    "about": 5.8, "across": 4.5, "against": 4.5, "along": 4.5,
    "among": 4.3, "around": 5.0, "behind": 4.5, "beside": 3.8,
    "beyond": 4.0, "despite": 4.0, "except": 4.0, "inside": 4.0,
    "outside": 4.0, "toward": 4.0, "towards": 4.0, "upon": 4.0,
    "within": 4.3, "without": 4.8, "enough": 4.8, "quite": 4.5,
    "rather": 4.3, "perhaps": 4.3, "whether": 4.3,
    "whose": 4.0, "wherever": 3.5, "whenever": 3.8, "whatever": 4.5,
    "whichever": 3.0, "whoever": 3.5,
    "another": 5.0, "either": 4.5, "neither": 3.8,
    "together": 4.8, "alone": 4.5, "apart": 3.8,
    "forward": 4.0, "backward": 3.3, "upward": 3.3, "downward": 3.3,
    "quickly": 4.3, "slowly": 4.0, "carefully": 4.0, "easily": 4.0,
    "simply": 4.0, "recently": 4.3, "finally": 4.5, "suddenly": 4.0,
    "especially": 4.3, "particularly": 4.0, "certainly": 4.3,
    "likely": 4.3, "obviously": 4.0, "clearly": 4.3,
    "apparently": 3.8, "typically": 3.8, "usually": 4.3,
    "normally": 3.8, "frequently": 3.5, "occasionally": 3.5,
    "immediately": 4.0, "eventually": 4.0, "ultimately": 3.8,
    "literally": 4.0, "completely": 4.3, "absolutely": 4.3,
    "entirely": 3.8, "mostly": 4.0, "partly": 3.5,

    # More nouns for coverage
    "answer": 4.5, "problem": 5.0, "solution": 4.3, "approach": 4.0,
    "method": 4.3, "technique": 3.8, "strategy": 3.8, "plan": 4.5,
    "goal": 4.3, "target": 4.0, "purpose": 4.0, "meaning": 4.3,
    "concept": 3.8, "theory": 3.8, "principle": 3.5, "rule": 4.3,
    "standard": 4.0, "requirement": 3.8, "condition": 4.0,
    "situation": 4.3, "circumstance": 3.5, "environment": 4.0,
    "experience": 4.5, "knowledge": 4.3, "skill": 4.0, "ability": 4.0,
    "chance": 4.3, "opportunity": 4.0, "choice": 4.3, "decision": 4.3,
    "opinion": 4.0, "view": 4.3, "thought": 5.2, "feeling": 4.3,
    "concern": 4.0, "interest": 4.3, "attention": 4.3, "focus": 4.0,
    "effort": 4.0, "attempt": 4.0, "success": 3.8, "progress": 4.0,
    "step": 4.5, "stage": 4.0, "phase": 3.8, "period": 4.0,
    "century": 3.8, "decade": 3.5, "future": 4.3, "past": 4.5,
    "present": 4.3, "age": 4.3, "generation": 3.8, "tradition": 3.5,
    "culture": 4.0, "society": 4.0, "country": 4.5, "nation": 4.0,
    "state": 4.0, "region": 3.8, "area": 4.8, "district": 3.5,
    "street": 4.3, "road": 4.3, "building": 4.3, "structure": 4.0,
    "material": 3.8, "resource": 4.0, "tool": 4.0, "equipment": 3.5,
    "machine": 3.8, "technology": 4.3, "science": 4.0, "nature": 4.0,
    "energy": 4.0, "light": 4.5, "sound": 4.3, "color": 4.0,
    "shape": 3.8, "form": 4.3, "space": 4.5, "ground": 4.3,
    "field": 4.3, "market": 4.3, "industry": 4.0, "economy": 3.8,
    "price": 4.3, "cost": 4.3, "rate": 4.3, "tax": 3.8,
    "bank": 4.0, "account": 4.3, "fund": 3.8, "investment": 3.8,
    "trade": 3.8, "product": 4.3, "brand": 3.8, "quality": 4.0,
    "design": 4.3, "style": 4.0, "pattern": 3.8, "model": 4.3,
    "image": 4.3, "picture": 4.3, "photo": 4.0, "film": 4.0,
    "movie": 4.3, "music": 4.5, "song": 4.3, "art": 4.3,
    "sport": 4.0, "game": 4.8, "race": 4.0, "match": 4.0,
    "season": 4.0, "weather": 3.8, "temperature": 3.5,
    "food": 4.5, "drink": 4.0, "meal": 3.8, "breakfast": 3.8,
    "lunch": 3.8, "dinner": 4.0, "restaurant": 3.8,
    "hospital": 3.8, "doctor": 4.0, "patient": 3.8, "health": 4.5,
    "disease": 3.5, "treatment": 3.8, "medicine": 3.8,
    "school": 4.5, "college": 4.0, "university": 4.0, "student": 4.3,
    "class": 4.5, "course": 4.0, "degree": 3.8, "professor": 3.5,
    "language": 4.3, "english": 4.0, "sentence": 3.8, "paragraph": 3.3,
    "chapter": 3.8, "section": 4.0, "article": 4.0, "document": 4.0,
    "letter": 4.0, "note": 4.3, "list": 4.5, "table": 4.3,
    "figure": 3.8, "chart": 3.5, "map": 3.8, "sign": 4.0,
    "symbol": 3.5, "character": 4.0, "digit": 3.3,
    "network": 4.0, "connection": 3.8, "relationship": 4.0,
    "communication": 3.8, "conversation": 4.0, "discussion": 3.8,
    "argument": 3.8, "debate": 3.5, "agreement": 3.8, "contract": 3.5,
    "policy": 4.0, "regulation": 3.5, "legislation": 3.3,
    "election": 3.5, "campaign": 3.5, "vote": 3.8,
    "police": 4.0, "court": 4.0, "judge": 3.8, "lawyer": 3.5,
    "crime": 3.8, "prison": 3.5, "army": 3.5, "military": 3.5,
    "attack": 4.0, "defense": 3.8, "protection": 3.5,
    "family": 5.0, "husband": 4.0, "wife": 4.0, "son": 4.3,
    "daughter": 4.0, "brother": 4.0, "sister": 4.0, "baby": 4.0,
    "animal": 4.0, "dog": 4.3, "cat": 4.0, "bird": 3.8,
    "fish": 3.8, "tree": 3.5, "flower": 3.5, "garden": 3.5,
    "park": 3.8, "forest": 3.5, "mountain": 3.5, "river": 3.8,
    "ocean": 3.5, "island": 3.5, "beach": 3.5, "stone": 3.5,
    "fire": 4.0, "sun": 3.8, "moon": 3.5, "star": 4.0,

    # More verbs
    "accept": 4.0, "achieve": 4.0, "acknowledge": 3.5, "acquire": 3.5,
    "adapt": 3.5, "adjust": 3.5, "admit": 4.0, "adopt": 3.5,
    "advance": 3.5, "advise": 3.5, "affect": 4.0, "afford": 3.8,
    "agree": 4.5, "aim": 3.8, "allow": 4.5, "announce": 3.8,
    "apologize": 3.5, "apply": 4.0, "appreciate": 3.8,
    "argue": 3.8, "arrange": 3.5, "arrive": 4.0, "assume": 4.0,
    "attach": 3.5, "attract": 3.5, "avoid": 4.0,
    "base": 4.0, "beat": 4.0, "belong": 3.8, "blame": 3.5,
    "block": 3.8, "blow": 3.8, "borrow": 3.5, "bother": 3.5,
    "break": 4.8, "broke": 4.0, "broken": 4.0,
    "breathe": 3.5, "burn": 3.8, "celebrate": 3.5,
    "challenge": 3.8, "charge": 3.8, "chase": 3.5,
    "choose": 4.3, "chose": 3.8, "chosen": 3.5,
    "claim": 4.0, "clean": 4.0, "climb": 3.5, "collect": 3.8,
    "combine": 3.5, "compare": 3.8, "compete": 3.5, "complain": 3.5,
    "complete": 4.3, "concentrate": 3.3, "concern": 4.0,
    "connect": 3.8, "consist": 3.5, "construct": 3.3,
    "contain": 4.0, "continue": 4.5, "contribute": 3.5,
    "control": 4.3, "cook": 3.8, "copy": 3.8, "correct": 4.0,
    "cover": 4.0, "cross": 3.8, "cry": 3.8,
    "damage": 3.8, "dance": 3.5, "deal": 4.3, "deliver": 3.8,
    "demand": 3.8, "deny": 3.5, "depend": 3.8, "describe": 4.0,
    "deserve": 3.5, "destroy": 3.8, "detect": 3.5,
    "determine": 4.0, "disappear": 3.5, "discover": 4.0,
    "discuss": 4.0, "divide": 3.5, "doubt": 3.8, "drag": 3.5,
    "draw": 4.0, "drew": 3.5, "drawn": 3.3, "dream": 3.8,
    "dress": 3.5, "drink": 4.0, "drive": 4.3, "drove": 3.5, "driven": 3.3,
    "drop": 4.0, "earn": 3.8, "eat": 4.3, "ate": 3.5, "eaten": 3.3,
    "employ": 3.3, "enable": 3.8, "encourage": 3.8, "enjoy": 4.0,
    "ensure": 4.0, "enter": 4.0, "escape": 3.5, "establish": 3.8,
    "estimate": 3.5, "evaluate": 3.5, "examine": 3.5, "exchange": 3.5,
    "exist": 4.0, "expand": 3.8, "explore": 3.8, "express": 3.8,
    "extend": 3.5, "fail": 4.0, "feed": 3.5, "fight": 4.0,
    "figure": 3.8, "fill": 4.0, "finish": 4.3, "fit": 4.0,
    "fly": 4.0, "flew": 3.5, "focus": 4.0, "force": 4.3,
    "forget": 4.3, "forgot": 3.8, "forgive": 3.5,
    "gain": 3.8, "gather": 3.5, "grab": 3.8, "grow": 4.3, "grew": 3.8,
    "guarantee": 3.5, "guess": 4.0, "guide": 3.8,
    "hang": 3.8, "hate": 4.0, "hide": 3.8, "hit": 4.3,
    "hope": 4.5, "hurt": 4.0, "identify": 3.8, "ignore": 3.8,
    "imagine": 4.0, "improve": 4.0, "include": 4.5,
    "increase": 4.0, "indicate": 3.8, "influence": 3.5,
    "inform": 3.5, "insist": 3.5, "inspire": 3.5,
    "intend": 3.5, "introduce": 3.8, "investigate": 3.5,
    "invite": 3.8, "involve": 4.0, "judge": 3.8,
    "jump": 3.8, "kick": 3.5, "knock": 3.8, "land": 3.8,
    "laugh": 4.0, "launch": 3.8, "lay": 3.8, "lead": 4.3,
    "led": 3.8, "lean": 3.5, "lie": 4.0, "lift": 3.8,
    "limit": 3.8, "link": 4.0, "listen": 4.3,
    "lock": 3.5, "manage": 4.0, "mark": 3.8,
    "matter": 4.5, "measure": 3.8, "mention": 4.0,
    "mind": 4.8, "miss": 4.3, "mix": 3.5, "notice": 4.0,
    "obtain": 3.5, "occur": 3.8, "offer": 4.3, "order": 4.3,
    "organize": 3.5, "overcome": 3.5, "owe": 3.5,
    "perform": 4.0, "permit": 3.3, "pick": 4.0,
    "plan": 4.5, "point": 5.0, "pour": 3.5, "practice": 4.0,
    "prepare": 4.0, "present": 4.3, "press": 3.8,
    "pretend": 3.5, "prevent": 3.8, "produce": 4.0,
    "promise": 3.8, "protect": 3.8, "prove": 4.0,
    "provide": 4.5, "publish": 3.8, "purchase": 3.5,
    "pursue": 3.3, "push": 3.8, "raise": 4.0,
    "react": 3.5, "realize": 4.0, "receive": 4.0,
    "recognize": 3.8, "record": 4.0, "recover": 3.5,
    "refer": 3.8, "reflect": 3.5, "refuse": 3.8,
    "relate": 3.5, "release": 3.8, "rely": 3.5,
    "repeat": 3.8, "replace": 3.8, "reply": 3.8,
    "represent": 3.8, "request": 4.3, "rescue": 3.3,
    "resolve": 3.5, "respond": 3.8, "reveal": 3.8,
    "ride": 3.8, "ring": 3.8, "rise": 4.0, "rose": 3.8,
    "rush": 3.5, "satisfy": 3.3, "score": 3.8,
    "settle": 3.8, "shake": 3.8, "share": 4.3,
    "shoot": 3.8, "shot": 4.0, "shout": 3.5, "shut": 3.8,
    "sing": 3.8, "sleep": 4.0, "slide": 3.5, "smell": 3.5,
    "smile": 4.0, "solve": 3.8, "spread": 3.8,
    "steal": 3.5, "stick": 3.8, "strike": 3.5,
    "struggle": 3.8, "study": 4.5, "suffer": 3.5,
    "supply": 3.5, "support": 4.3, "suppose": 4.0,
    "surprise": 3.8, "surround": 3.3, "survive": 3.5,
    "suspect": 3.5, "swing": 3.5, "switch": 3.8,
    "teach": 4.0, "taught": 3.8, "test": 4.5,
    "throw": 4.0, "threw": 3.5, "tie": 3.5,
    "touch": 4.0, "train": 3.8, "travel": 4.0,
    "treat": 4.0, "trust": 4.0, "visit": 4.0,
    "vote": 3.8, "wake": 3.8, "walk": 4.3,
    "warn": 3.8, "wash": 3.5, "waste": 3.8, "wear": 4.0,
    "wore": 3.5, "wonder": 4.0, "worry": 4.0, "wish": 4.3,
    "wrap": 3.3, "yell": 3.5,
}

# Common bigrams with log-probability scores
# Higher score = more likely to co-occur
SEED_BIGRAMS: dict[str, float] = {
    # Determiner + noun/adjective
    "the_same": 5.5, "the_first": 5.3, "the_other": 5.3,
    "the_most": 5.3, "the_best": 5.0, "the_world": 5.0,
    "the_new": 4.8, "the_next": 4.8, "the_only": 4.8,
    "the_right": 4.8, "the_old": 4.5, "the_last": 4.8,
    "the_great": 4.3, "the_whole": 4.3, "the_end": 4.5,
    "a_lot": 5.5, "a_few": 5.3, "a_little": 5.0,
    "a_new": 4.8, "a_good": 4.8, "a_big": 4.5,
    "a_great": 4.5, "a_long": 4.5, "a_single": 4.0,

    # Pronoun + verb
    "i_am": 6.0, "i_have": 5.8, "i_think": 5.8, "i_want": 5.5,
    "i_can": 5.5, "i_would": 5.3, "i_will": 5.3, "i_need": 5.3,
    "i_know": 5.5, "i_was": 5.5, "i_do": 5.3, "i_like": 5.0,
    "i_just": 5.0, "i_don": 5.0, "i_feel": 4.8, "i_see": 4.8,
    "i_love": 4.8, "i_mean": 4.8, "i_got": 4.8,
    "you_can": 5.5, "you_are": 5.5, "you_have": 5.3,
    "you_want": 5.3, "you_know": 5.5, "you_will": 5.0,
    "you_need": 5.0, "you_think": 4.8, "you_should": 4.8,
    "you_would": 4.8, "you_like": 4.5, "you_could": 4.8,
    "you_get": 4.5, "you_do": 4.8, "you_see": 4.5,
    "it_is": 6.0, "it_was": 5.8, "it_would": 5.0,
    "it_can": 4.8, "it_will": 4.8, "it_has": 4.8,
    "he_was": 5.5, "he_had": 5.3, "he_is": 5.0,
    "he_said": 5.3, "he_would": 4.8,
    "she_was": 5.3, "she_had": 5.0, "she_said": 5.0, "she_is": 4.8,
    "we_have": 5.3, "we_can": 5.0, "we_are": 5.3,
    "we_need": 4.8, "we_will": 4.8, "we_should": 4.5,
    "they_are": 5.3, "they_have": 5.0, "they_were": 4.8,
    "they_will": 4.5, "they_can": 4.5, "they_would": 4.3,

    # Verb + preposition/particle
    "go_to": 5.5, "go_back": 4.8, "go_out": 4.5,
    "come_back": 5.0, "come_from": 4.8, "come_to": 4.8,
    "come_in": 4.5, "come_out": 4.5, "come_up": 4.5,
    "get_to": 5.0, "get_out": 4.5, "get_up": 4.5,
    "get_back": 4.5, "get_in": 4.3, "get_into": 4.0,
    "look_at": 5.3, "look_for": 5.0, "look_like": 4.8,
    "look_up": 4.0, "look_out": 3.8, "look_into": 3.8,
    "take_a": 4.8, "take_the": 4.5, "take_it": 4.5,
    "take_care": 4.3, "take_off": 4.0, "take_out": 4.0,
    "make_a": 5.0, "make_it": 5.0, "make_the": 4.8,
    "make_sure": 4.8, "make_up": 4.0,
    "give_me": 4.8, "give_it": 4.5, "give_up": 4.3,
    "turn_out": 4.3, "turn_around": 3.8, "turn_off": 3.8,
    "put_it": 4.5, "put_on": 4.3, "put_in": 4.0,
    "set_up": 4.3, "set_out": 3.8,
    "pick_up": 4.3, "pick_out": 3.5,
    "keep_up": 4.0, "keep_it": 4.0, "keep_on": 3.8,
    "find_out": 4.8, "find_a": 4.5,
    "work_on": 4.8, "work_with": 4.5, "work_out": 4.3,
    "work_for": 4.3, "work_in": 4.0,
    "think_about": 5.0, "think_of": 4.8, "think_that": 4.5,
    "talk_about": 4.8, "talk_to": 4.8,
    "tell_me": 5.0, "tell_you": 4.8,
    "ask_for": 4.5, "ask_me": 4.3,
    "help_me": 5.0, "help_you": 4.5, "help_with": 4.3,
    "wait_for": 4.3, "wait_a": 3.8,
    "try_to": 5.0, "want_to": 5.5, "need_to": 5.3,
    "have_to": 5.5, "going_to": 5.5, "used_to": 5.0,
    "able_to": 4.8, "like_to": 4.5, "start_to": 4.0,

    # Adjective + noun
    "good_thing": 4.3, "new_york": 4.8, "long_time": 4.8,
    "first_time": 5.0, "last_time": 4.5, "same_time": 4.5,
    "next_time": 4.3, "real_time": 4.0, "right_now": 4.8,
    "right_thing": 4.0, "little_bit": 4.5, "young_man": 3.8,
    "old_man": 4.0, "big_deal": 3.8, "good_idea": 4.0,
    "whole_thing": 4.0, "best_way": 4.0, "other_people": 4.3,
    "other_hand": 4.3, "each_other": 5.0,
    "few_things": 4.3, "open_source": 3.8,

    # Common phrases
    "of_the": 6.5, "in_the": 6.5, "to_the": 6.0,
    "on_the": 6.0, "at_the": 5.8, "for_the": 5.8,
    "with_the": 5.5, "from_the": 5.3, "by_the": 5.3,
    "of_a": 5.5, "in_a": 5.5, "to_a": 5.0, "for_a": 5.0,
    "as_a": 5.0, "with_a": 5.0, "on_a": 5.0,
    "is_a": 5.5, "was_a": 5.3, "is_the": 5.3, "was_the": 5.0,
    "that_is": 5.0, "that_the": 5.0, "that_was": 4.8,
    "this_is": 5.3, "there_is": 5.3, "there_are": 5.0,
    "there_was": 5.0, "here_is": 4.5,
    "one_of": 5.3, "some_of": 5.0, "all_of": 5.0,
    "most_of": 4.8, "many_of": 4.5, "part_of": 4.8,
    "kind_of": 4.8, "sort_of": 4.3, "lot_of": 5.0,
    "out_of": 5.3, "because_of": 4.5, "instead_of": 4.3,
    "more_than": 5.0, "less_than": 4.3, "rather_than": 4.0,
    "as_well": 4.8, "as_much": 4.3, "as_if": 4.3,
    "not_only": 4.3, "not_just": 4.3,
    "do_you": 5.5, "do_not": 5.5, "did_not": 5.3,
    "does_not": 5.0, "would_not": 4.8, "could_not": 4.8,
    "should_not": 4.5, "can_not": 4.5, "will_not": 4.5,
    "has_been": 5.0, "have_been": 5.0, "had_been": 4.8,
    "will_be": 5.0, "would_be": 5.0, "could_be": 4.8,
    "should_be": 4.5, "may_be": 4.5, "might_be": 4.3,
    "must_be": 4.3, "can_be": 4.8,
    "going_on": 4.5, "based_on": 4.5,
    "up_to": 4.8, "due_to": 4.5, "according_to": 4.3,
    "back_to": 4.8, "next_to": 4.0,
    "such_as": 4.5, "so_much": 4.3, "so_many": 4.0,
    "too_much": 4.3, "how_to": 5.0, "what_is": 5.0,
    "what_do": 4.8, "what_are": 4.5, "how_do": 4.8,
    "how_much": 4.3, "how_many": 4.0, "who_is": 4.3,
    "where_is": 4.3, "when_is": 3.8,

    # Tech-specific bigrams
    "source_code": 3.8, "open_source": 3.8,
    "data_base": 3.5, "web_site": 3.5,
    "pull_request": 3.5, "code_review": 3.3,
    "test_case": 3.3, "error_message": 3.3,
    "user_interface": 3.3, "command_line": 3.3,
    "file_system": 3.3, "operating_system": 3.0,
    "machine_learning": 3.5, "deep_learning": 3.3,
    "neural_network": 3.0, "data_set": 3.3,
    "real_world": 4.0, "high_level": 3.5,
    "long_term": 3.8, "short_term": 3.5,
    "at_least": 4.8, "at_all": 4.8,

    # AI conversation bigrams
    "can_you": 5.5, "could_you": 5.0, "would_you": 5.0,
    "should_i": 4.5, "can_i": 5.0, "could_i": 4.3,
    "let_me": 5.0, "help_me": 5.0,
    "thank_you": 5.3, "you_please": 4.5,
    "me_know": 4.3, "me_help": 4.0, "me_work": 4.0,
    "please_help": 4.3, "please_tell": 3.8,
    "want_me": 4.0, "need_me": 3.8,
    "tell_me": 5.0, "show_me": 4.5,
    "me_the": 4.8, "me_a": 4.5,
}


def build_extended_words(seed: dict[str, float]) -> dict[str, float]:
    """Extend seed words with morphological variants."""
    extended = dict(seed)

    # Common suffixes to generate variants
    suffix_rules = [
        # (base_ending, suffix, freq_penalty)
        ("", "s", 0.3),       # plurals
        ("", "ed", 0.3),      # past tense
        ("", "ing", 0.3),     # gerund
        ("", "er", 0.4),      # comparative / agent
        ("", "est", 0.5),     # superlative
        ("", "ly", 0.3),      # adverb
        ("", "ness", 0.5),    # noun from adj
        ("", "ment", 0.5),    # noun from verb
        ("", "tion", 0.5),    # noun from verb
        ("", "able", 0.5),    # adjective
        ("", "ible", 0.5),    # adjective
        ("e", "ing", 0.3),    # drop e + ing
        ("e", "ed", 0.3),     # drop e + ed (though often same)
    ]

    for word, freq in list(seed.items()):
        if len(word) < 3:
            continue
        for ending, suffix, penalty in suffix_rules:
            if ending and word.endswith(ending):
                variant = word[:-len(ending)] + suffix
            else:
                variant = word + suffix
            if variant not in extended and len(variant) <= 20:
                extended[variant] = max(freq - penalty, 1.0)

    return extended


def build_extended_bigrams(
    seed_bigrams: dict[str, float],
    words: dict[str, float],
) -> dict[str, float]:
    """Extend seed bigrams with high-frequency word pairs."""
    extended = dict(seed_bigrams)

    # Add bigrams for very common word pairs not already covered
    top_words = sorted(words.items(), key=lambda x: -x[1])[:50]
    for i, (w1, f1) in enumerate(top_words):
        for w2, f2 in top_words[i + 1:]:
            key = f"{w1}_{w2}"
            rev_key = f"{w2}_{w1}"
            if key not in extended and rev_key not in extended:
                score = (f1 + f2) / 4.0  # rough co-occurrence estimate
                if score > 2.5:
                    extended[key] = round(score, 2)

    return extended


def main() -> None:
    output_dir = Path(__file__).parent.parent / "src" / "txtpand" / "corpus"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Building word corpus...")
    words = build_extended_words(SEED_WORDS)
    print(f"  {len(words)} words generated")

    # Sort by frequency descending for easier inspection
    words_sorted = dict(sorted(words.items(), key=lambda x: (-x[1], x[0])))

    words_path = output_dir / "words.json"
    with open(words_path, "w") as f:
        json.dump(words_sorted, f, indent=None, separators=(",", ":"))
    print(f"  Written to {words_path}")

    print("Building bigram corpus...")
    bigrams = build_extended_bigrams(SEED_BIGRAMS, words)
    print(f"  {len(bigrams)} bigrams generated")

    bigrams_sorted = dict(sorted(bigrams.items(), key=lambda x: (-x[1], x[0])))

    bigrams_path = output_dir / "bigrams.json"
    with open(bigrams_path, "w") as f:
        json.dump(bigrams_sorted, f, indent=None, separators=(",", ":"))
    print(f"  Written to {bigrams_path}")

    print("Done!")


if __name__ == "__main__":
    main()
