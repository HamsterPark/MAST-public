"""
Monkey-patch for nanonis_spm v1.0.9 parseGeneralResponse() bug.

The package uses two format string orderings for response types:
  - "*+c", "*-c" (star first) — handled by the original parser
  - "+*c", "+*i", "+*b" (plus/minus first) — NOT handled, causes struct.error

The "+*X" format is actually more common (53 uses vs 36 for "*+c"), but the
parser only checks ResponseType[0] == '*'. This patch adds handling for the
"+*X" / "-*X" case as self-contained prepended/non-prepended arrays.

Also patched: ``Nanonis.send`` itself — see ``_patched_send`` for the two
"can never return" defects it fixes (审计 致命三).

Also patched: ``decodeArray`` / ``decodeArrayPrepended`` — every numeric array
field came back as a list of 1-TUPLES because ``struct.unpack``'s return value
was appended without being unwrapped. See ``_patched_decodeArray``
(KNOWN_ISSUES §2.21).

Also patched, two defects that are NOT the same and do NOT affect the same
functions — the distinction matters, so they are listed apart:

* ``decodeStringPrepended`` ONLY: its 4-byte length was decoded by concatenating
  the bytes' DECIMAL digits, which equals the true length only while the top
  three bytes are zero — right for every length ≤ 255 by coincidence, wrong for
  every length ≥ 256 (256 → 10, 300 → 144, 1000 → 3232; 44 of the lengths in
  0–299 are wrong).
* ``decodeStringPrepended`` AND ``decodeSingularString``: neither bounds-checked
  its byte reads, so a misaligned reply raised ``IndexError`` and ``parseError``
  never ran — replacing the instrument's own error text with a Python traceback.

``decodeSingularString`` takes its length as an ARGUMENT and never decodes one,
so the first defect cannot reach it or the ``*-c`` replies it serves. An earlier
version of this docstring said otherwise; it was wrong, and it would have sent
the next reader looking for a length bug on the ``Scan.FrameDataGrab`` path that
was never there. See ``_patched_decodeStringPrepended`` (KNOWN_ISSUES §2.29).

See: https://github.com/AlanGG12/nanonis_spm (upstream repo)
"""

from __future__ import annotations

import struct
import numpy as np
from nanonis_spm import Nanonis


# Save reference to original method
_original_parseGeneralResponse = Nanonis.parseGeneralResponse
_original_send = Nanonis.send
_original_decodeArray = Nanonis.decodeArray
_original_decodeArrayPrepended = Nanonis.decodeArrayPrepended
_original_decodeStringPrepended = Nanonis.decodeStringPrepended
_original_decodeSingularString = Nanonis.decodeSingularString


# ── send() hardening constants ───────────────────────────────────────────
#: The timeout nanonis_spm v1.0.9 unconditionally writes onto the socket after
#: every successful round-trip (``NanonisClass.py:171``). Any pre-call timeout
#: at or above this is assumed to be that leak rather than a caller's intent.
_LIB_BOGUS_TIMEOUT_S = 1000.0

#: Fallback recv timeout used when the socket has no usable one (blocking mode,
#: or poisoned by an unpatched call). ``ConnectionPool`` overrides this from
#: ``NanonisConfig.timeout_s`` so patch and pool agree.
_default_recv_timeout_s = 5.0


def set_default_recv_timeout(seconds: float) -> None:
    """Set the fallback recv timeout ``_patched_send`` restores.

    Called by :class:`mast.core.connection.ConnectionPool` on (re)connect so
    the patch's fallback matches ``NanonisConfig.timeout_s`` instead of a
    hardcoded 5 s. Ignores nonsense values — a bad config must never be able
    to reintroduce the unbounded wait this patch exists to remove.
    """
    global _default_recv_timeout_s
    try:
        val = float(seconds)
    except (TypeError, ValueError):
        return
    if 0.0 < val < _LIB_BOGUS_TIMEOUT_S:
        _default_recv_timeout_s = val


def _restore_timeout(conn) -> None:
    """Put the socket timeout back to the caller's value after a round-trip.

    nanonis_spm ends every ``send`` with ``settimeout(1000)``, so the 5 s recv
    timeout ``ConnectionPool.connect_all`` sets only ever applies to the FIRST
    command on a socket; from the second onward a host that stops answering
    without sending FIN (frozen / unplugged / powered off) stalls the caller
    for up to ~17 minutes per command. It also breaks the comms circuit
    breaker, whose "consecutive" streak window is 30 s
    (``comms_health._STREAK_WINDOW_S``): two 1000 s timeouts are never within
    30 s of each other, so ``_streak`` stays 1 and the breaker never OPENs.
    """
    try:
        prev = conn.gettimeout()
    except Exception:  # noqa: BLE001 — a fake/duck-typed socket in tests
        return
    if prev is None or prev <= 0 or prev >= _LIB_BOGUS_TIMEOUT_S:
        prev = _default_recv_timeout_s
    try:
        conn.settimeout(prev)
    except Exception:  # noqa: BLE001
        pass


def _recv_exact(conn, n: int) -> bytes:
    """Read exactly ``n`` bytes, or raise. Never spins.

    Replaces the upstream body loop

        while (Recv_BodySize != len(Recv_Body) or counter < 1000):
            Recv_Body += self.connection.recv(Recv_BodySize - len(Recv_Body))

    which has two defects, both fatal on a real instrument:

    * **EOF spins forever.** After the peer closes (Nanonis killed, cable
      pulled, OS sends FIN) ``recv`` returns ``b''`` *immediately* — it does
      not block, so the socket timeout never fires. The length can never catch
      up, so the loop burns 100 % of a core in pure Python and **never returns
      and never raises**. Measured 2026-07-28: the thread was still alive
      after 8 s with no exception. Every caller of that role's lock then
      queues behind it forever (see ``ConnectionPool.safe_call``).
    * **``or counter < 1000``.** Even a perfectly complete response runs 1000
      extra ``recv(0)`` syscalls before the loop can exit.

    Here EOF becomes ``ConnectionAbortedError`` — a ``ConnectionError``, which
    ``safe_call`` already catches as a TCP-level failure, so it drives the
    existing reconnect + circuit-breaker path instead of hanging.
    """
    if n <= 0:
        return b""
    chunks: list[bytes] = []
    got = 0
    while got < n:
        chunk = conn.recv(n - got)
        if not chunk:
            raise ConnectionAbortedError(
                f"Nanonis peer closed the connection mid-response "
                f"({got}/{n} bytes received)"
            )
        chunks.append(chunk)
        got += len(chunk)
    return b"".join(chunks)


def _patched_send(self, Command, Body, BodyType):
    """Hardened replacement for ``nanonis_spm.Nanonis.send``.

    Request framing is byte-for-byte the upstream logic (the ``handle*`` /
    ``correctType`` helpers are the class's own). Three changes, all on the
    wire I/O:

    1. ``sendall`` instead of ``send`` — a short write would truncate the
       32-byte-command frame and desynchronise the stream on a port that has
       to be restarted by hand to recover.
    2. header and body read via :func:`_recv_exact` — EOF raises instead of
       spinning, and a fragmented header no longer feeds a short slice to
       ``struct.unpack``.
    3. the caller's socket timeout is restored on the way out instead of
       being overwritten with 1000 s (:func:`_restore_timeout`).
    """
    BodyPart = bytearray()

    # ── request framing: verbatim from nanonis_spm v1.0.9 ────────────────
    for i in range(0, len(Body)):
        if "*" in BodyType[i]:
            instance = Body[i]
            body_type = BodyType[i]
            if "c" in BodyType[i]:
                if isinstance(Body[i], str):
                    # Array of chars (i.e. string)
                    BodyPart = self.handleString(Body[i], BodyType[i], BodyPart)
                else:
                    # Array of strings
                    BodyPart = self.handleArrayString(Body[i], BodyType[i], BodyPart)
            elif "-" in BodyType[i]:
                for j in range(0, len(Body[i])):
                    instance[j] = self.correctType(body_type[2], instance[j])
                    Body[i] = instance
                BodyPart = self.handleArray(Body[i], BodyType[i], BodyPart)
            elif "+" in BodyType[i]:
                for j in range(0, len(Body[i])):
                    instance[j] = self.correctType(body_type[2], instance[j])
                    Body[i] = instance
                BodyPart = self.handleArrayPrepend(Body[i], BodyType[i], BodyPart)
            else:
                BodyPart = self.handleArray(Body[i], BodyType[i], BodyPart)
        else:
            if "2" in BodyType[i]:
                BodyPart = self.handle2DArray(Body[i], BodyType[i], BodyPart)
            else:
                Body[i] = self.correctType(BodyType[i], Body[i])
                BodyPart = BodyPart + struct.pack('>' + BodyType[i], Body[i])

    SendResponseBack = True
    BodySize = len(BodyPart)
    ZeroBuffer = bytearray(2)

    Message = bytearray(str(Command).ljust(32, '\0').encode()) + \
              BodySize.to_bytes(4, byteorder='big') + \
              SendResponseBack.to_bytes(2, byteorder='big') + \
              ZeroBuffer + \
              BodyPart
    if self.displayInfo == 1:
        print('Send message: ')
        print(Message)

    conn = self.connection
    # ── wire I/O: hardened ───────────────────────────────────────────────
    try:
        sendall = getattr(conn, "sendall", None)
        if sendall is not None:
            sendall(Message)
        else:  # duck-typed transport in tests
            conn.send(Message)

        Recv_Header = _recv_exact(conn, 40)  # header — always 40 bytes
        Recv_BodySize = struct.unpack('>I', Recv_Header[32:36])[0]
        Recv_Body = _recv_exact(conn, Recv_BodySize)
    finally:
        # ALWAYS, including on timeout/EOF: leaving 1000 s behind on a socket
        # we are about to reconnect is how one dead call poisons the next.
        _restore_timeout(conn)

    Recv_Command = Recv_Header[0:32].decode().strip('0').replace('\x00', '')
    if self.displayInfo == 1:
        print("BodySize:", Recv_BodySize)
        print("Received Body:", len(Recv_Body))
        print('Received data:')
        print(Recv_Header)
        print(Recv_Body)

    if Recv_Command == Command:
        if self.displayInfo == 1:
            print('Correct Command.')
        return Recv_Body
    else:
        print('Wrong Command')
        return []


# ── 数组解包：元素没有从 struct.unpack 的元组里取出来 ─────────────────────────
#
# ``decodeArray`` (NanonisClass.py:186-196) 和 ``decodeArrayPrepended``
# (:224-236) 是同一个写法：
#
#     decoded_num = struct.unpack('>' + responseType, decoded_num)
#     decoded_nums.append(decoded_num)      # ← unpack 返回元组，原样 append
#
# ``struct.unpack`` **永远**返回元组。所以受影响的不是某一个动词，是**每一个数值
# 数组字段** —— ``*i`` ``*I`` ``*f`` ``*d``（走 Prepended）与 ``**f`` ``**I``
# ``**i``（走 decodeArray）全部回来是 ``[(0,), (30,)]`` 而不是 ``[0, 30]``。
#
# 下游需要标量数组，直接对单元素元组执行 int/float 会失败。
# 解包应在共同解析入口完成，避免每个调用方分别维护转换逻辑。

# 落点选在这里而不是逐处补，理由有三：
#
# 1. **这个模块已经替换了 ``parseGeneralResponse``**，而它自己的 ``+*X`` / ``-*X``
#    分支写的是 ``result.append(val[0])`` —— 解对了。同一个模块里两条数组路径一条
#    解包一条不解包，本身就是要消掉的不一致。
# 2. 那十余处只是把数组**原样透传**进报告的地方（``datalog.py``、
#    ``user_output.py``、``marks.py``、``readback.py`` 等），逐处补根本够不着 ——
#    它们没有解包代码可补，只会把 ``[(0,), (24,)]`` 印给模型看。
# 3. 逐处补就是在造第七、第八份实现。
#
# **反向风险已逐个核对**：那六份实现全都写成「是元组就取 [0]，否则直接用」
# （``isinstance(v, tuple)`` / ``len(x) == 1`` 型判断），以及
# ``io/nanonis_files.py`` 的 ``scalar_int()``；两种形态都接受，所以补丁之后走的是
# 「否则」那一支，仍然正确。这一条不能靠假定 —— 六份都看过了。
#
# **调用方那层双形态容错防的是一个不会静默发生的场景 —— 别照着它再加。**
# 第一版把它写成「防补丁没生效」，不准确。``apply()`` 是模块末尾**直接调用**的
# 纯赋值，而 ``core/connection.py`` 无条件 import 本模块；``Nanonis`` 一旦导入不了、
# 或上游改掉了 ``decodeArray`` 这个名字，本模块顶部的
# ``_original_decodeArray = Nanonis.decodeArray`` 当场 ``AttributeError``，整个进程
# 起不来。**要么生效，要么响亮地失败 —— 没有中间态。**
#
# 所以调用侧留着容错的真实理由只剩两条，都不是「防缺陷」：测试里 conftest 把
# nanonis_spm 换成 MagicMock、桩给什么就是什么；以及 ``revert()`` 期间。
# 它属于纵深防御，**不该排在任何真缺陷前面**。
#
# **只动两条分支，字符串两支一个字都别改。** parser 的数组分支有四条，走四个不同
# 函数，而 bug 只在其中两条：
#
#     ``*+c`` → decodeStringPrepended   没有 bug —— 逐字符 chr() 拼，不调 unpack
#     ``*-c`` → decodeSingularString    没有 bug —— 同上
#     ``**X`` → decodeArray             有
#     ``*X``  → decodeArrayPrepended    有
#
# 字符串数组与数值数组经过不同解码分支，不应未经区分地统一处理。
# 使用合成协议字节对 ``parseGeneralResponse`` 验证各分支，见
# ``tests/v2/unit/core/test_nanonis_patch_decode_array.py``。该测试验证解析语义，
# 不依赖仪器模块是否已加载，也不声称某台仪器发出过该字节流。


def _patched_decodeArray(self, response, index, numOfElements, responseType):
    """``decodeArray`` with the 1-tuple actually unpacked.

    Two changes from upstream:

    1. ``struct.unpack(...)[0]`` — the fix. See the module comment above.
    2. the stride comes from ``struct.calcsize`` instead of a hardcoded ``4``.
       This is a no-op for every ``**X`` spec nanonis_spm actually ships
       (``**f`` ``**I`` ``**i`` — all 4 bytes), but it removes a live
       inconsistency: ``_patched_parseGeneralResponse`` **already** advances the
       outer byte counter by ``calcsize`` for this branch, so a future ``**d``
       would have had a correct counter and a ``struct.error`` here.
    """
    if isinstance(numOfElements, list):
        return []
    fmt = '>' + responseType
    size = struct.calcsize(fmt)
    decoded_nums = []
    for _ in range(0, numOfElements):
        decoded_nums.append(struct.unpack(fmt, response[index:(index + size)])[0])
        index += size
    return decoded_nums


def _patched_decodeArrayPrepended(self, response, index, numOfElements,
                                  responseType):
    """``decodeArrayPrepended`` with the 1-tuple actually unpacked.

    Only the unwrap changes. The ``8 if 'd' else 4`` stride is kept verbatim
    from upstream **on purpose**: ``parseGeneralResponse`` advances its own byte
    counter for this branch by exactly that rule, so switching one side to
    ``calcsize`` would desynchronise the two for any 2-byte spec. Neither side
    is exercised today (no ``*h`` / ``*H`` exists in the library); keeping them
    identical means there is nothing to get wrong later.
    """
    increment = 8 if responseType == 'd' else 4
    if isinstance(numOfElements, list):
        return []
    fmt = '>' + responseType
    decoded_nums = []
    for _ in range(0, numOfElements):
        decoded_nums.append(
            struct.unpack(fmt, response[index:(index + increment)])[0])
        index += increment
    return decoded_nums


# ── 字符串解码：仪器的错误原文被换成一个 IndexError ──────────────────────────
#
# 字符串解析失败会掩盖仪器原始错误信息，必须同时处理长度和边界。

# 两个独立的毛病，都在 ``decodeStringPrepended`` (NanonisClass.py:198-214)：
#
# 1. **长度根本不是按大端解的。** 它把 4 个字节各自的**十进制字面量拼起来**再
#    ``int()``：``int(str(b0)+str(b1)+str(b2)+str(b3))``。前三字节为 0 时
#    ``"0"+"0"+"0"+str(n)`` 恰好等于 n —— 所以长度 ≤255 一直是对的，**纯属巧合**。
#    **≥256 一律错**，不是「256 这一个特例」：0–299 里有 44 个长度是错的，
#    256 → 10、300 → 144、1000 → 3232（后者还会连带越界抛 IndexError）。
#    这个区别决定下一个人要不要回头查历史数据。
# 2. **``response[index + i]`` 是标量索引，没有边界检查** —— 越界即 ``IndexError``。
#
# 当错误段被误当成普通数据读取时，错误文本中的字节可能被解成巨大长度，
# 导致越界并掩盖真实诊断。错误信息应透传，解析失败不能冒充仪器的拒绝原因。
#
# 修法刻意保守，**不改任何目前能正常工作的东西**：
#
# * 长度改成 ``struct.unpack('>i')`` —— 对 ≤255 与那个巧合**逐位相同**，只有
#   ≥256 从「静默错」变成「对」。
# * 读之前夹紧到 ``len(response)``，越界就截断/停下，让 ``parseError`` 有机会
#   跑到、把仪器原文交出来。
# * 解码用 **latin-1**，因为上游是逐字节 ``chr(response[i])`` —— 逐字节一字符。
#   换成 utf-8 会让多字节字符的 ``len(str) < len(bytes)``，而调用方正是用
#   ``counter += 4 + len(item)`` 推进字节计数器的，**那会把后面所有字段错位**。


def _patched_decodeStringPrepended(self, response, index, numOfStrings):
    """``decodeStringPrepended`` with a real big-endian length and bounds checks.

    Well-formed replies decode byte-for-byte identically to upstream; a garbled
    or misaligned one degrades to short/empty strings instead of ``IndexError``,
    so the instrument's own error text still reaches the operator.
    """
    decoded_strings = []
    end_of_body = len(response)
    for _ in range(0, numOfStrings):
        if index + 4 > end_of_body:
            break
        str_len = struct.unpack('>i', response[index:index + 4])[0]
        index += 4
        if str_len <= 0:
            decoded_strings.append("")
            continue
        stop = min(index + str_len, end_of_body)
        # latin-1: one byte -> one char, matching upstream's chr(). Keeps
        # len(str) == len(bytes) so the caller's counter arithmetic holds.
        decoded_strings.append(response[index:stop].decode('latin-1'))
        index += str_len
        if stop < index:  # ran off the end — nothing left to read
            break
    return decoded_strings


def _patched_decodeSingularString(self, response, index, stringLength):
    """``decodeSingularString``, bounds-checked. Same ``chr()``-equivalent
    latin-1 decoding; same reason (``*-c`` is on the scan hot path via
    ``Scan.FrameDataGrab``, where a misparse must not become an ``IndexError``).
    """
    if stringLength <= 0:
        return ""
    stop = min(index + stringLength, len(response))
    if stop <= index:
        return ""
    return response[index:stop].decode('latin-1')


#: 一次 Nanonis 回包体的末尾恒定是「错误段」：错误状态(int32) + 描述长度(int32)
#: + 描述(bytes)。前面才是各命令自己声明的返回字段。
_ERROR_SECTION_HEADER = 8

#: 「回包是真的，是我们的解码规格不对」这句话的开头。**给这个区分起了名字**，
#: 因为它要被另一个模块认出来（``monitoring.pump.Osci2TProbe``），而按字面量各写
#: 一份的话，改一处就会静默失配 —— 那正是 2026-08-09 那类缺陷的形状。
LAYOUT_MISMATCH_PREFIX = "response layout mismatch"


def _salvage_rejected_reply(Response, ResponseTypes, exc) -> list:
    """声明的字段读越界之后，退回来按「只有错误段」重读一遍回包。

    Nanonis 拒绝一条命令时（模块没加载、动词不认识、参数不对），回包体里
    **只有错误段** —— 命令自己声明的那些返回字段一个都不在。而
    `parseGeneralResponse` 是**先读声明字段、最后才读错误段**的，于是它拿
    `["i","i","*f"]` 去读一段 `[status][size][desc]`：前两个 int32 读成了
    「状态」和「描述长度」，第三个字段于是按「描述长度」个 float32 去读，
    一路读出缓冲区 → `struct.error`。

    后果不是「报错慢了一点」，是**仪器自己说的那句话被吃掉了**：
    `NeedModule` 这个字符串永远到不了调用方，而全仓判「模块在不在」靠的正是
    它（`pump._guard`、`zburst._guard` 都写着 `if "NeedModule" in err`）。
    于是一次「模块没开」被报成一句读不懂的 struct 错误。

    salvage 之后，两种情况**被分开**，这正是这个函数存在的全部理由：

    * `status != 0` → 这确实是一次**拒绝**。返回仪器自己的描述文本，
      `NeedModule` 那条判据重新生效；
    * `status == 0` → body 的开头**不是**错误段，说明这是一份**真实回包**，
      而我们照样读越界了 ⇒ **声明的 ResponseTypes 与实际布局不符**。
      这时返回一句点名 spec 的话，而不是假装模块不在。

    换句话说：**机器自己回答「是模块没开，还是我们的 spec 写错了」**，
    而不是让调用方从同一种 struct 异常猜测原因。

    返回值保持 `[ErrorString, Response, Variables]` 三元形状，调用方不必知道
    这里发生过什么。
    """
    body = bytes(Response or b"")
    if len(body) < _ERROR_SECTION_HEADER:
        return [f"reply too short to parse ({len(body)} bytes): {exc}", body, []]
    text = _error_only_text(body)
    if text is not None:
        return [text, body, []]
    # 不是「只有错误段」的回包 ⇒ 回包是真的，越界的是我们的解码规格。说清楚，
    # 否则下一个人会照 2026-08-09 那样，把它误读成「模块没加载」。
    return [
        f"{LAYOUT_MISMATCH_PREFIX}: declared {list(ResponseTypes)} overran a "
        f"{len(body)}-byte reply that is not an error-only body (i.e. the "
        f"instrument answered normally) — the ResponseTypes for this verb are "
        f"wrong ({exc})",
        body, [],
    ]




def _reply_ends_in_a_clean_error_section(body: bytes, counter: int) -> bool:
    """counter 处是不是一个 **status=0** 的错误段（= 仪器说「一切正常」）？

    错误段的形状是 ``[4B status][4B desc_len][desc_len B text]``，位于回包末尾。
    ``status == 0`` 时它就是「无错误」的标记，而且此时 ``desc_len`` 通常是 0，
    整段正好 8 字节。

    要求**长度恒等式成立**（counter + 8 + desc_len == len(body)），不能只看
    status：如果 counter 落错了位置，那 8 个字节可能碰巧是一对小整数，
    而恒等式把这种巧合挡在外面。
    """
    tail = len(body) - int(counter)
    if tail < _ERROR_SECTION_HEADER:
        return False
    try:
        status, desc_len = struct.unpack(
            '>ii', body[counter:counter + _ERROR_SECTION_HEADER])
    except struct.error:
        return False
    if status != 0 or desc_len < 0:
        return False
    return tail == _ERROR_SECTION_HEADER + desc_len

def _looks_like_a_real_error(body: bytes, counter: int, text: str) -> bool:
    """``parseError`` 解出来的这段，是仪器真的在报错，还是我们读进了数据段？

    判据与 :func:`_error_only_text` 同源 —— **长度恒等式**，不是「有没有非零
    状态字」。错误段的形状是 ``status(i4) + desc_len(i4) + desc``，所以从
    ``counter`` 起的剩余字节数必须**正好**是 ``8 + desc_len``。

    为什么不用「文本可不可打印」这类判据：那会朝两边都出错。真实的错误描述里
    可能带非 ASCII，而一段 float64 数据也可能碰巧全是可打印字符 —— 用一个会
    朝两边都出错的判据去分辨两件相反的事，正是本文件反复记录的那类失败。
    """
    tail = len(body) - int(counter)
    if tail < _ERROR_SECTION_HEADER:
        return False
    try:
        status, desc_len = struct.unpack(
            '>ii', body[counter:counter + _ERROR_SECTION_HEADER])
    except struct.error:
        return False
    if status == 0 or desc_len < 0:
        return False
    # 恒等式：剩下的正好是「头 + 描述」，一个字节都不多
    return tail == _ERROR_SECTION_HEADER + desc_len

def _error_only_text(body: bytes):
    """body 是不是一份「只有错误段」的回包？是就返回它的描述文本，否则 ``None``。

    判据是长度恒等式，而不是把 body 的前四字节当错误状态：正常回包的
    首字段也可能非零，错误的字段规格会把它误读成拒绝。

    只有错误段时 body 长度必须恰好为 ``8 + 描述长度``。前面还有正常声明
    字段的回包不满足这项结构约束。
    """
    if len(body) < _ERROR_SECTION_HEADER:
        return None
    status, desc_len = struct.unpack('>ii', body[:_ERROR_SECTION_HEADER])
    if status == 0 or desc_len < 0:
        return None
    if len(body) != _ERROR_SECTION_HEADER + desc_len:
        return None
    desc = body[_ERROR_SECTION_HEADER:]
    # 描述为空也要说话：一个空的错误串会被 safe_call 判成「没出错」。
    return desc.decode('utf-8', errors='replace').strip() or \
        f"Nanonis error status {status}"


def _patched_parseGeneralResponse(self, Response, ResponseTypes):
    """Patched version that handles +*c / +*i / +*b / -*X format strings.

    外面还包了一层 :func:`_salvage_rejected_reply` —— 读越界时不要把仪器自己的
    错误文本连同异常一起丢掉。见那个函数的 docstring。
    """
    try:
        out = _parse_general_response_strict(self, Response, ResponseTypes)
    except (struct.error, UnicodeDecodeError) as exc:
        return _salvage_rejected_reply(Response, ResponseTypes, exc)
    # 读越界只是拒绝的**一种**表现，不是全部。描述短到能被声明字段吃下去时，
    # 严格解析会「成功」，而 ``parseError`` 从一个越过末尾的位置切片、拿到空串
    # —— 于是**一次拒绝被报成一次成功**，调用方拿走一串垃圾数值。
    # （实测：8 字节的空描述拒绝 + ``["i","i","*f"]`` 正是这样，一个异常都不抛。）
    # 所以成功路径上也要问一句：这份 body 是不是压根就只有错误段。
    if isinstance(out, list) and len(out) >= 3 and not out[0] and ResponseTypes:
        text = _error_only_text(bytes(Response or b""))
        if text is not None:
            return [text, out[1], []]
    return out


def _parse_general_response_strict(self, Response, ResponseTypes):
    """严格按 ResponseTypes 解码；读不动就抛（由上面那层接住并分诊）。"""
    counter = 0
    Variables = []
    universalLength = 0

    for ResponseType in ResponseTypes:
        # --- PATCH: handle +*X and -*X format strings ---
        if (len(ResponseType) >= 3
                and ResponseType[0] in ('+', '-')
                and ResponseType[1] == '*'):
            elem_type = ResponseType[2]
            if ResponseType[0] == '+':
                if elem_type == 'c':
                    # Self-contained prepended string: 4-byte length + string data
                    #
                    # utf-8 HERE and latin-1 in _patched_decodeStringPrepended is
                    # not an oversight — don't "unify" them. The difference is
                    # which number advances the byte counter:
                    #   +*c (here) counter += str_len         (declared BYTES)
                    #   *+c        counter += Variables[-2]   (declared BYTES)
                    #   **c        counter += 4 + len(item)   (decoded CHARS) ←
                    # Only the LAST one feeds a decoded length back into the byte
                    # counter, and that is the one decodeStringPrepended serves
                    # for Marks.PointsGet. There, utf-8 would make
                    # len(str) < len(bytes) on any non-ASCII byte and shift every
                    # following field; here it is free to produce real unicode.
                    str_len = struct.unpack('>i', Response[counter:counter+4])[0]
                    counter += 4
                    string_val = Response[counter:counter+str_len].decode(
                        'utf-8', errors='replace')
                    counter += str_len
                    Variables.append(string_val)
                else:
                    # Self-contained prepended array: 4-byte count + data
                    arr_len = struct.unpack('>i', Response[counter:counter+4])[0]
                    counter += 4
                    elem_size = struct.calcsize('>' + elem_type)
                    result = []
                    for i in range(arr_len):
                        val = struct.unpack(
                            '>' + elem_type,
                            Response[counter:counter+elem_size])
                        result.append(val[0])
                        counter += elem_size
                    Variables.append(result)
            else:  # '-'
                # Non-prepended: length from previous variable.
                # Guard (): if this `-*X` spec is the FIRST response
                # type, Variables is empty and Variables[-1] would raise
                # IndexError; if the preceding variable isn't an int count
                # (e.g. a string/array), range(arr_len) would raise TypeError.
                # Either case means a malformed/misaligned response — treat the
                # length as 0 (empty array) rather than crashing the parser,
                # matching the "no elements" outcome the original code yields
                # when the prior count is 0. Patch semantics for a well-formed
                # response (int count precedes the array) are unchanged.
                arr_len = Variables[-1] if Variables else 0
                # Accept Python ints and numpy integer scalars (a prior
                # +*i array yields Python ints; numpy ints can appear via
                # reshaped/decoded counts). Anything else (str/float/array)
                # means misalignment → length 0.
                if isinstance(arr_len, (int, np.integer)) and not isinstance(arr_len, bool):
                    arr_len = int(arr_len)
                else:
                    arr_len = 0
                if arr_len < 0:
                    arr_len = 0
                elem_size = struct.calcsize('>' + elem_type)
                result = []
                for i in range(arr_len):
                    val = struct.unpack(
                        '>' + elem_type,
                        Response[counter:counter+elem_size])
                    result.append(val[0])
                    counter += elem_size
                Variables.append(result)
            continue

        # --- ORIGINAL LOGIC (unchanged) ---
        if ResponseType[0] != '*':
            if ResponseType[0] == '2':
                NoOfRows = Variables[-2]
                NoOfCols = Variables[-1]
                SentArray = []
                Datasize = struct.calcsize('>' + ResponseType[1])
                for i in range(NoOfRows * NoOfCols):
                    Value = struct.unpack(
                        '>' + ResponseType[1],
                        Response[counter:(counter + Datasize)])
                    counter = counter + Datasize
                    SentArray.append(Value)
                Variables.append(np.reshape(SentArray, (NoOfRows, NoOfCols)))
                if self.displayInfo == 1:
                    print(ResponseType, '  : ',
                          np.reshape(SentArray, (NoOfRows, NoOfCols)))
            else:
                Datasize = struct.calcsize('>' + ResponseType)
                Value = struct.unpack(
                    '>' + ResponseType,
                    Response[counter:(counter + Datasize)])
                Variables.append(Value[0])
                counter = counter + Datasize
        else:
            if ResponseType[1] == '+':
                # ═══════════════════════════════════════════════════════════
                # `*+c` 与 `*+i` 的**前置字段个数不一样**，不能共用一支
                # ═══════════════════════════════════════════════════════════
                #
                # 上游 v1.0.9（以及本 patch 的第一版）只看 ResponseType[1]=='+'
                # 就走字符串路径，于是 `*+i` 被 decodeStringPrepended 解析，
                # 随后 `counter + Variables[-2]` 里的 Variables[-2] 是上一个
                # `*+c` 留下的**字符串列表** ⇒
                #     TypeError: unsupported operand type(s) for +: 'int' and 'list'
                #
                # 若该分支解析失败，Scan_PropsGet 的参数清单就不可用；调用方为避免
                # 覆盖未知配置不会下发 Scan_PropsSet，连续扫描因此可能保持开启。
                # 解析分支必须按声明类型区分字符串与数值数组。
                #
                # 两者的声明形状（Scan.PropsGet 文档）：
                #   ... i(bytes) i(count) `*+c`   → 前面**两个** int，
                #                                   counter 前进 bytes
                #   ... i(count)          `*+i`   → 前面**一个** int，
                #                                   counter 前进 count*elem
                elem_type = ResponseType[2] if len(ResponseType) >= 3 else 'c'
                if elem_type != 'c':
                    n = Variables[-1]
                    if not isinstance(n, int) or n < 0:
                        n = 0        # 前一个字段不是计数 ⇒ 回包已经错位，别再乘下去
                    elem_size = struct.calcsize('>' + elem_type)
                    result = []
                    for _i in range(n):
                        result.append(struct.unpack(
                            '>' + elem_type,
                            Response[counter:counter + elem_size])[0])
                        counter += elem_size
                    Variables.append(result)
                else:
                    NoOfChars = Variables[-1]
                    String = self.decodeStringPrepended(
                        Response, counter, NoOfChars)
                    step = Variables[-2] if len(Variables) >= 2 else 0
                    if not isinstance(step, int):
                        # 声明里本该是字节数的位置放着别的东西 ⇒ 用实际解码出的
                        # 长度兜底，让后续字段至少有机会对齐，而不是当场抛。
                        step = sum(4 + len(x) for x in (String or []))
                    counter = counter + step
                    Variables.append(String)
            elif ResponseType[1] == '-':
                NoOfChars = Variables[-1]
                String = self.decodeSingularString(
                    Response, counter, NoOfChars)
                counter = counter + NoOfChars
                Variables.append(String)
            elif ResponseType[1] == '2' and len(ResponseType) >= 3                     and ResponseType[2] == 'c':
                # *2c 表示由 rows、cols 指定尺寸的二维前置长度字符串数组。
                # Scan.PropsGet 的一维模块名称与二维参数表不能共用字节数解释：
                # 前者由字节数和数量描述，后者由行列数描述。
                # 二维分支从前两个字段取行列数，逐串读取长度，并按实际消费字节推进。
                # 若误把行数当字节数，错误解析器会停在数据段内部，把参数内容当错误信息。
                _rc = []
                for _v in (Variables[-2] if len(Variables) >= 2 else None,
                           Variables[-1] if Variables else None):
                    if isinstance(_v, (int, np.integer)) and not isinstance(_v, bool)                             and int(_v) >= 0:
                        _rc.append(int(_v))
                    else:
                        # 前置字段不是一对计数 ⇒ 回包已经错位，别再乘下去
                        _rc = [0, 0]
                        break
                _rows, _cols = _rc[0], _rc[1]
                # 每个元素至少占 4 字节（长度前缀），乘出来超过剩余长度就是错位
                if _rows * _cols * 4 > max(0, len(Response) - counter):
                    _rows = _cols = 0
                _grid = []
                for _r in range(_rows):
                    _row = []
                    for _c in range(_cols):
                        if counter + 4 > len(Response):
                            break
                        (_sl,) = struct.unpack('>i', Response[counter:counter + 4])
                        counter += 4
                        if _sl < 0 or counter + _sl > len(Response):
                            counter -= 4
                            break
                        _row.append(Response[counter:counter + _sl].decode(
                            'utf-8', errors='replace'))
                        counter += _sl
                    _grid.append(_row)
                Variables.append(_grid)
                if self.displayInfo == 1:
                    print(ResponseType, '  : ', _grid)
            elif ResponseType[1] == '*':
                universalLength = Variables[0]
                if ResponseType[2] == 'c':
                    Result = self.decodeStringPrepended(
                        Response, counter, universalLength)
                    if len(Result) != 0:
                        for item in Result:
                            counter = counter + 4 + len(item)
                else:
                    Result = self.decodeArray(
                        Response, counter, universalLength, ResponseType[2])
                    # PATCH: counter advance must respect element size — `d`
                    # (float64) is 8 bytes, not 4. Upstream nanonis_spm v1.0.9
                    # hardcodes `* 4` which makes any subsequent fields
                    # misaligned for `**d` / `**i`-typed arrays.
                    elem_size = struct.calcsize('>' + ResponseType[2])
                    counter = counter + (universalLength * elem_size)
                Variables.append(Result)
            else:
                Result = self.decodeArrayPrepended(
                    Response, counter, Variables[-1], ResponseType[1])
                if ResponseType[1] == 'd':
                    increment = 8
                else:
                    increment = 4
                if Variables[-1] != 0:
                    counter = counter + (Variables[-1] * increment)
                Variables.append(Result)

    ErrorString = self.parseError(Response, counter)
    if len(ErrorString) != 0 and _reply_ends_in_a_clean_error_section(
            bytes(Response or b""), counter):

        # counter 正好位于 status=0 且长度一致的错误段时，仪器没有报告错误。
        # 应优先采用此处完整错误段，避免尾部倒推算法把正常数据误认成错误描述。
        return ["", Response, Variables]
    if len(ErrorString) != 0 and not _looks_like_a_real_error(
            bytes(Response or b""), counter, ErrorString):
        # 返回字段规格与布局不符时，counter 可能落入数据段，普通数值字节会被误读成错误。
        # 此时保留已解析的 Variables，并明确报告规格不匹配，不能冒充仪器拒绝命令。
        return [
            "%s: parseError at offset %d produced %d bytes that are not an "
            "error section (the reply is %d bytes and does not satisfy "
            "8+desc_len) — the declared ResponseTypes %s are misaligned, the "
            "instrument answered normally"
            % (LAYOUT_MISMATCH_PREFIX, counter, len(ErrorString),
               len(bytes(Response or b"")), list(ResponseTypes)),
            Response, Variables,
        ]
    if len(ErrorString) != 0:
        print('The following error appeared:', "\n", ErrorString)
        return [ErrorString, Response, Variables]
    else:
        if self.displayInfo == 1:
            print('No error messages. Error status was: 0')
        return [ErrorString, Response, Variables]


def _patched_SpectrumAnlzr_DataGet(self, Spectrum_Analyzer_instance):
    """nanonis_spm v1.0.9 ships `["f", "f", "i", "*f"]` as the return spec
    for SpectrumAnlzr.DataGet. Per the Nanonis Programming Interface manual
    (section "SpectrumAnlzr.DataGet"), the response actually contains:

        Data f0   : float64
        Data df   : float64
        Data Y size : int32
        Data Y    : 1D array of float64

    With the wrong spec the parser advances the byte counter by 4 per
    float64 sample (off by 4 bytes) and eventually trips a UnicodeDecodeError
    while trying to decode the "error description" string at a misaligned
    offset. Fix: declare float64 throughout.
    """
    return self.quickSend(
        "SpectrumAnlzr.DataGet",
        [Spectrum_Analyzer_instance], ["i"],
        ["d", "d", "i", "*d"],
    )


def _patched_Osci1T_TimebaseGet(self):
    """nanonis_spm v1.0.9 ships a copy-paste bug: `Osci1T_TimebaseGet`
    sends the command name **"Osci1T.TimebaseSet"** (the SETTER) instead of
    "Osci1T.TimebaseGet". With the wrong command the RT controller either
    rejects the message (TimebaseSet expects a Timebase-index argument that
    isn't sent) or returns nothing useful, so the available-timebases list a
    caller needs to choose a sample rate is never delivered.

    Per the Nanonis Programming Interface manual ("Osci1T.TimebaseGet"), the
    response is:

        Timebase index   : int32   (index of the selected timebase)
        Number of timebases : int32
        Timebases (s)    : 1D array of float32 (timebase values in seconds)

    Fix: send the correct command with the documented return spec. The
    available timebases depend on the RT frequency and RT oversampling.
    """
    return self.quickSend("Osci1T.TimebaseGet", [], [], ["i", "i", "*f"])


# ── Osci2T (dual-channel scope): five copy-paste bugs ────────────────────────
#
# nanonis_spm v1.0.9 wired most of the Osci2T family to the WRONG command name.
# Verified by reading the shipped source (NanonisClass.py:9279-9366):
#
#   Osci2T_ChSet        -> sends "Osci1T.ChSet"        (and with TWO arguments,
#                                                       which the 1-channel
#                                                       command does not take)
#   Osci2T_ChGet        -> sends "Osci1T.ChGet"
#   Osci2T_TimebaseSet  -> sends "Osci1T.TimebaseSet"
#   Osci2T_TimebaseGet  -> sends "Osci1T.TimebaseSet"  (the SETTER, same bug the
#                                                       Osci1T getter has)
#   Osci2T_OversamplGet -> sends "Osci2T.OversamplSet" (the SETTER)
#
# The consequence is not a clean failure: a caller that "configures Osci2T"
# silently reconfigures the OTHER oscilloscope — the one the tunnelling-current
# monitor is pumping. That is why these are fixed together with the dual-channel
# recorder rather than left for later.
#
# The channel verb is RESOLVED, NOT GUESSED. Nanonis names multi-channel setters
# in the plural ("SignalChart.ChsSet" takes channel A + channel B, and the
# Osci2T signature likewise takes two), which makes "Osci2T.ChsSet" the likely
# real name — but the shipped docstring says "Osci2T.ChSet", the TCP protocol
# manual is not part of this repo, and no simulator implements the module. A
# wrong guess here fails silently on the real machine. So we probe, and we probe
# with the GETTER: it takes no arguments, so a wrong name costs one rejected
# read instead of a misconfigured scope.

#: Candidate (verb-stem, getter response spec) pairs, most likely first.
_OSCI2T_CH_CANDIDATES: tuple[tuple[str, list], ...] = (
    ("Chs", ["i", "i"]),     # plural, two channels — matches SignalChart.ChsSet
    ("Ch", ["i", "i"]),      # what the shipped docstring claims
    ("Ch", ["i"]),           # ...if it really is a single-channel reply
)


def _osci2t_ch_stem(self):
    """Resolved ("Chs"|"Ch", response-spec), probing once per client."""
    cached = getattr(self, "_mast_osci2t_ch", None)
    if cached is not None:
        return cached
    last = _OSCI2T_CH_CANDIDATES[-1]
    for stem, spec in _OSCI2T_CH_CANDIDATES:
        try:
            rv = self.quickSend(f"Osci2T.{stem}Get", [], [], spec)
        except Exception:  # noqa: BLE001 — a wire error says nothing about the name
            return last
        if isinstance(rv, (list, tuple)) and rv and not rv[0]:
            self._mast_osci2t_ch = (stem, spec)
            return self._mast_osci2t_ch
    # Nothing answered. Cache nothing — the module may simply not be loaded yet
    # (it is licensed but its front panel has to be open), and a later attempt
    # should get a fresh probe rather than inherit this verdict.
    return last



#: ``Scan.PropsGet`` 的响应规格。与 nanonis_spm v1.0.9 的唯一差别是**最后一个
#: 字符串数组用 ``*2c``**（2D，rows×cols）而不是 ``*+c``（1D，bytes+count）。
#:
#: 位置与命名两处独立来源一致：nanonis_spm NanonisClass.py 的 ResponseTypes 表，
#: 与协议手册的字段表。二者只在**最后那个数组的维度**上分歧 —— 手册说它是 2D
#: 且前面是 (rows, cols)，上游的记号却是 1D 的 ``*+c``（前面该是 bytes+count）。
#: 二维分支必须按行列数和实际字符串长度推进。
_SCAN_PROPS_GET_SPEC: list[str] = [
    "I", "I", "I",          # continuous / bouncy / autosave
    "i", "*-c",             # series name
    "i", "*-c",             # comment
    "i", "i", "*+c",        # modules names: bytes, count, 1D strings
    "i", "*+i",             # per-module parameter counts: count, 1D ints
    "i", "i", "*2c",        # parameters: rows, cols, **2D** strings
    "I",                    # autopaste
]


def _patched_Scan_PropsGet(self):
    """Scan.PropsGet —— 用修正过的响应规格（最后一个数组是 2D）。"""
    return self.quickSend("Scan.PropsGet", [], [], _SCAN_PROPS_GET_SPEC)

def _patched_Osci2T_ChsGet(self):
    """Read the two channels Osci2T is displaying. Verb resolved by probing."""
    stem, spec = _osci2t_ch_stem(self)
    return self.quickSend(f"Osci2T.{stem}Get", [], [], spec)


def _patched_Osci2T_ChsSet(self, ChannelAIndex, ChannelBIndex):
    """Set both Osci2T channels. Verb resolved via the read-only getter probe,
    so an unknown name is discovered without ever writing to the wrong scope."""
    stem, _spec = _osci2t_ch_stem(self)
    return self.quickSend(f"Osci2T.{stem}Set",
                          [ChannelAIndex, ChannelBIndex], ["i", "i"], [])


def _patched_Osci2T_TimebaseSet(self, TimebaseIndex):
    """v1.0.9 sends "Osci1T.TimebaseSet" — i.e. retimes the OTHER scope, the one
    the current monitor is pumping. Body spec is unchanged (uint16 index)."""
    return self.quickSend("Osci2T.TimebaseSet", [TimebaseIndex], ["H"], [])

# Osci2T.TimebaseGet 保留兼容候选响应规格。
# 库声明使用 uint16 索引，另一兼容形式使用 int32；不得仅凭 Osci1T 的形状类推。
# 通过声明数量与数组长度、索引范围等自洽检查选择能解释回包的布局，
# 规格不符必须报告，不能静默采用猜测。
_OSCI2T_TIMEBASE_SPECS: tuple[list, ...] = (
    ["H", "i", "*f"],
    ["i", "i", "*f"],
)


def _timebase_reply_is_coherent(decoded: list) -> bool:
    """一份时基回包解出来自不自洽 —— **传输层**的判据，只回答「这个 spec 解对了吗」。

    与 `monitoring.pump.check_timebase_table()` 刻意分开，两者问的不是一个问题：
    那一个是**面向用户**的三态结论（ok / mismatch / unverified，会上报到
    `/api/monitoring/status`），说的是「这张表可不可信」；这一个只在**选 spec**
    时用，答案只有真假，而且必须在 pump 拿到数据**之前**就有。
    （层级上也过不去：`core` 不能 import `monitoring`。）
    """
    if len(decoded) < 3:
        return False
    index, count, values = decoded[0], decoded[1], decoded[2]
    if not isinstance(values, (list, tuple)):
        return False
    try:
        index, count = int(index), int(count)
    except (TypeError, ValueError):
        return False
    if count <= 0 or count != len(values):
        return False
    return 0 <= index < count


def _patched_Osci2T_TimebaseGet(self):
    """v1.0.9 sends "Osci1T.TimebaseSet" — the setter, of the wrong scope.

    命令名的修复是**有据的**（读库源码即可确认）；响应规格**没有**，所以这里探，
    见 :data:`_OSCI2T_TIMEBASE_SPECS`。探到的那条按客户端缓存，之后直接用。
    """
    cached = getattr(self, "_mast_osci2t_tb_spec", None)
    if cached is not None:
        return self.quickSend("Osci2T.TimebaseGet", [], [], cached)
    last = None
    for spec in _OSCI2T_TIMEBASE_SPECS:
        rv = self.quickSend("Osci2T.TimebaseGet", [], [], list(spec))
        last = rv
        if not (isinstance(rv, (list, tuple)) and len(rv) >= 3):
            continue
        if not rv[0] and _timebase_reply_is_coherent(list(rv[2])):
            self._mast_osci2t_tb_spec = list(spec)
            return rv
        # **刻意不在这里提前返回**，即使 rv[0] 里有一句像模像样的错误。
        # 用错的规格去读一份**真实**回包时，salvage 会把回包的头 4 个字节当成
        # 「错误状态」—— 那个数几乎必然非 0（它其实是时基索引），于是一份好端端
        # 的回包被报成一次拒绝。提前返回就等于:第一个候选猜错 ⇒ 第二个永远没
        # 机会。多发一次只读命令的代价，远小于「因为顺序不对而永久退回 1T」。
    # 一条都不自洽：不缓存（下次重新探），把最后一次的结果如实交回去。
    # 真的是拒绝时，每个候选都会解出同一段错误描述（错误段在最前面，与声明的
    # 字段无关），所以"最后一次"带回来的就是仪器自己那句话。
    return last if last is not None else ("", b"", [])


def _patched_Osci2T_OversamplGet(self):
    """v1.0.9 sends "Osci2T.OversamplSet" — the setter — with no argument."""
    return self.quickSend("Osci2T.OversamplGet", [], [], ["H"])


# ── 协议里有、库里没有的绑定（2026-08-04） ──────────────────────────────────
#
# 两个技能一直在调根本不存在的方法，所以**永远返回失败**：
#   * ``SetSessionPath``      → ``Util_SessionPathSet``
#   * ``PLLPerfectUpdateZTC`` → ``PLL_PerfectPLLUpdtZTC``
#
# 判定过程值得写下来，因为「协议缺失」和「绑定缺失」的处理方式相反（前者删技能、
# 后者补绑定），而两者从库这一侧看起来一模一样：
#
#   * ``Util.SessionPathSet``     —— 协议 p.275 有。库里只有 ``Get`` 没有 ``Set``，
#     而 Util 模块其他每一对都是全的（AcqPeriod / RTFreq / RTOversampl /
#     SettingsLoad-Save / LayoutLoad-Save）。**孤零零一个 Get 是绑定缺失的特征。**
#   * ``PLL.PerfectPLLUpdtZTC``   —— 协议 p.187 有。这个从命名规律上看不出来
#     （73 个 ``PLL_*`` 里没有一个叫 PerfectPLL），差点被当成「协议里也没有」而
#     删掉技能 —— 直到查了 ``TCP-reference/tcp_protocol.txt``。
#     **手头的 Nanonis 软件手册（CHM）里没有 TCP 协议章节，用它查会得到假阴性。**
#
# 参数格式照抄库里结构相同的那条，不自己发明：
#   * SessionPathSet  ← ``Util_SettingsSave``（同为 路径字符串 + uint32 开关）
#   * PerfectPLLUpdtZTC ← ``PLL_FreqShiftAutoCenter``（同为单个 Modulator index）
#
# 顺带记下：``PLL.PerfectPLLApply``（协议 p.187，参数同为 Modulator index (int)）
# **同样不在库里**。这里不补 —— 没有任何技能在用它。记在这里是为了下一个人不必
# 把上面这段查证再做一遍。

def _patched_Util_SessionPathSet(self, Session_path, Save_settings_to_previous):
    """Util.SessionPathSet — 设置 session 文件夹路径（协议 p.275）。

    Arguments:
      -- Session path size (int) + Session path (string)   → ``"+*c"``
      -- Save settings to previous (unsigned int32)         → ``"I"``
    Return arguments: 仅 Error。
    """
    return self.quickSend(
        "Util.SessionPathSet",
        [Session_path, Save_settings_to_previous],
        ["+*c", "I"], [],
    )


def _patched_PLL_PerfectPLLUpdtZTC(self, Modulator_index):
    """PLL.PerfectPLLUpdtZTC — 更新 Z 控制器时间常数（协议 p.187）。

    Arguments:
      -- Modulator index (int)，**有效值从 1 开始**（不是 0）  → ``"i"``
    Return arguments: 仅 Error。
    """
    return self.quickSend(
        "PLL.PerfectPLLUpdtZTC", [Modulator_index], ["i"], [],
    )


#: 本模块**新增**（而非覆盖）的方法。``revert`` 靠这张表把它们删干净 —— 覆盖类
#: 要还原成原实现，新增类没有"原实现"可还原。
_ADDED_METHODS: tuple[str, ...] = (
    "Osci2T_ChsGet", "Osci2T_ChsSet",
    "Util_SessionPathSet", "PLL_PerfectPLLUpdtZTC",
)


def apply():
    """Apply the monkey-patch. Safe to call multiple times."""
    Nanonis.parseGeneralResponse = _patched_parseGeneralResponse
    Nanonis.decodeArray = _patched_decodeArray
    Nanonis.decodeArrayPrepended = _patched_decodeArrayPrepended
    Nanonis.decodeStringPrepended = _patched_decodeStringPrepended
    Nanonis.decodeSingularString = _patched_decodeSingularString
    Nanonis.Util_SessionPathSet = _patched_Util_SessionPathSet
    Nanonis.PLL_PerfectPLLUpdtZTC = _patched_PLL_PerfectPLLUpdtZTC
    Nanonis.SpectrumAnlzr_DataGet = _patched_SpectrumAnlzr_DataGet
    Nanonis.Scan_PropsGet = _patched_Scan_PropsGet
    Nanonis.Osci1T_TimebaseGet = _patched_Osci1T_TimebaseGet
    Nanonis.Osci2T_ChsGet = _patched_Osci2T_ChsGet
    Nanonis.Osci2T_ChsSet = _patched_Osci2T_ChsSet
    # The singular spellings stay as aliases of the fixed pair: they are what
    # skills/builtins/optional_scopes.py already calls, and leaving them pointed
    # at Osci1T would keep that skill quietly reconfiguring the wrong scope.
    Nanonis.Osci2T_ChGet = _patched_Osci2T_ChsGet
    Nanonis.Osci2T_ChSet = _patched_Osci2T_ChsSet
    Nanonis.Osci2T_TimebaseSet = _patched_Osci2T_TimebaseSet
    Nanonis.Osci2T_TimebaseGet = _patched_Osci2T_TimebaseGet
    Nanonis.Osci2T_OversamplGet = _patched_Osci2T_OversamplGet
    Nanonis.send = _patched_send


_original_Scan_PropsGet = Nanonis.Scan_PropsGet
_original_SpectrumAnlzr_DataGet = Nanonis.SpectrumAnlzr_DataGet
_original_Osci1T_TimebaseGet = Nanonis.Osci1T_TimebaseGet
_original_Osci2T_ChGet = Nanonis.Osci2T_ChGet
_original_Osci2T_ChSet = Nanonis.Osci2T_ChSet
_original_Osci2T_TimebaseSet = Nanonis.Osci2T_TimebaseSet
_original_Osci2T_TimebaseGet = Nanonis.Osci2T_TimebaseGet
_original_Osci2T_OversamplGet = Nanonis.Osci2T_OversamplGet


def revert():
    """Revert to original method."""
    Nanonis.parseGeneralResponse = _original_parseGeneralResponse
    Nanonis.decodeArray = _original_decodeArray
    Nanonis.decodeArrayPrepended = _original_decodeArrayPrepended
    Nanonis.decodeStringPrepended = _original_decodeStringPrepended
    Nanonis.decodeSingularString = _original_decodeSingularString
    Nanonis.Scan_PropsGet = _original_Scan_PropsGet
    Nanonis.SpectrumAnlzr_DataGet = _original_SpectrumAnlzr_DataGet
    Nanonis.Osci1T_TimebaseGet = _original_Osci1T_TimebaseGet
    Nanonis.Osci2T_ChGet = _original_Osci2T_ChGet
    Nanonis.Osci2T_ChSet = _original_Osci2T_ChSet
    Nanonis.Osci2T_TimebaseSet = _original_Osci2T_TimebaseSet
    Nanonis.Osci2T_TimebaseGet = _original_Osci2T_TimebaseGet
    Nanonis.Osci2T_OversamplGet = _original_Osci2T_OversamplGet
    for attr in _ADDED_METHODS:
        if hasattr(Nanonis, attr):
            delattr(Nanonis, attr)
    Nanonis.send = _original_send


# Auto-apply on import
apply()


# ── parseError:错误串不该跟着前置字段的解析偏差走 ────────────────────────
_original_parseError = Nanonis.parseError


def _patched_parseError(self, response, index):
    """取出仪器自己的错误文本。**起点从回包尾部倒推,不用 ``counter``。**

    ## 原实现的问题

        margin = 8            # 4B 错误状态 + 4B 错误描述长度
        errorIndex = index + margin
        errorString = response[errorIndex:].decode()

    ``index`` 就是 ``parseGeneralResponse`` 里那个 ``counter`` —— **前面所有
    返回值解析完之后的位置**。任何一个前置字段的长度算错(而这个库的长度算错
    正是本文件在修的那一族),``counter`` 就偏,错误串跟着从错的位置开始读,
    **开头被啃掉**。

    前置字段长度偏差会导致错误文本缺失开头，进而误导诊断。
    错误起点不能依赖可能已经偏移的 counter。

    ## 新实现

    错误段是 ``[4B status][4B len][len B text]`` 且位于回包**末尾**,所以
    从尾部倒推:末 4+n 字节里的那个 n 必须正好等于剩余文本长度 —— 对上了才用,
    对不上就退回原实现(**不猜**:宁可给出可能缺字的文本,也不给一段瞎解的字节)。
    """
    try:
        if isinstance(response, (bytes, bytearray)) and len(response) >= 8:
            # 末 4 字节之前的那个 int32 若正好等于其后文本长度,即命中
            for text_len in (len(response) - 8, ):
                if text_len <= 0:
                    break
                n = struct.unpack('>i', response[-text_len - 4:-text_len])[0]
                if n == text_len:
                    return response[-text_len:].decode(errors="replace")
            # 一般情形:扫描尾部可能的长度前缀位置
            for cut in range(1, min(len(response) - 4, 4096)):
                n = struct.unpack('>i', response[-cut - 4:-cut])[0]
                if n == cut:
                    return response[-cut:].decode(errors="replace")
    except Exception:  # noqa: BLE001 —— 解不出就退回原实现,绝不因此抛
        pass
    return _original_parseError(self, response, index)


Nanonis.parseError = _patched_parseError
