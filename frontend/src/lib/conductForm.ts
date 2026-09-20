// ════════════════════════════════════════════════════════════════════════════
// conductForm.ts —— 新建一份 conduct 时那张表单的**纯**判断。
//
// 与 `lib/conduct.ts` 的分工:那份管**已经存在**的 conduct(状态、闸、停滞);
// 这份管**还不存在**的那一份 —— 模板选哪个、每个格子填什么、填错了怎么说。
//
// ── 这个文件为什么存在 ──────────────────────────────────────────────────────
//
// 在它之前,「新建一份 conduct」只有 `curl -XPOST /api/conducts` 一条路,而空态
// 上写着的正是这句话。这与 approve 是**同一个形状**:设计 §7 让 approve 只接受
// UI 来源,于是不给按钮 approve 就不存在;新建的后端一直是齐的,不给表单新建也
// 一样不存在。做不到的动作与不存在的动作,在屏幕上长得一模一样。
//
// ── 四条铁律,每条都是一次事故 ───────────────────────────────────────────────
//
// **① 一个物理量默认值都不预填。** 模板的 `ParamSpec.default` 全是 `None`,而且
// 是刻意的:填不出来就说明这次实验的工作点还没定,那正是该停下来的时候。所以
// `ParamSpecRow` 契约里根本没有 `default` 这个字段(后端刻意不回)——渲染件拿不到
// 那个数就不会预填,这是「移除诱因」而不是「说服自己别填」。`sts_condition` 是最
// 刺眼的一例:出厂只有一个 `default` 组而**它刻意是未标定的**,预填 "default"
// 会让人以为工作点已经定了。
//
// **② 超包络拒绝，绝不夹紧。** 单位或指数错误必须明确拒绝；
// 截断到边界会隐藏原输入与实际工作点之间的差异。
//
// **③ 「读不到」不是一个值。** 模板列表拉不到时说的是「拉不到模板列表」,不是
// 「没有模板」。这两句话不一样:前者是故障,后者是事实。
//
// **④ 服务端校验是真源,这里只是提前告知。** 所以本地判断**不禁用提交按钮**:
// 一个本地判错就再也提交不上去的表单,是「能停不能解」的又一例。本地发现的问题
// 红字写在格子下面,提交照走,后端的 findings 回来之后逐条盖到对应格子上。
//
// ── 数字怎么进来 ────────────────────────────────────────────────────────────
//
// 输入永远是一段文字,所以**一定**有一个解析器,问题只是它有多诚实。
// `parseFloat("1.5nA")` 会给出 `1.5` —— 一比一复现上面那次事故。所以这里是**整串
// 严格匹配**:看不懂就报错,绝不截一半。顺带接受 Nanonis 面板的写法(`1.5n`、
// `5nm`),因为那正是用户手上那台机器的写法;前缀表由测试对着
// `mast/core/si_quantity.py` 的 `SI_PREFIXES` 对账。
//
// 但**前缀在这里不是强制的**,与后端 agent 通道相反 —— `si_quantity` 的 docstring
// 写得很清楚:强制前缀是给「模型供给的量」用的,「用户自己的 UI」不该付那个
// 代价。这里换成另一道防线:**把系统实际收到的那个数原样回读给人看**
// (`1.50 nA（= 1.5e-9 A）`),量级错了当场看得见。
// ════════════════════════════════════════════════════════════════════════════

import { fmtSI } from "./units.ts";

// ── SI 前缀 ──────────────────────────────────────────────────────────────────

/**
 * 前缀 → 十进制指数。镜像 `mast/core/si_quantity.py` 的 `SI_PREFIXES`,由
 * `frontend/test/conductForm.test.ts` 直接读那个 .py 对账。
 *
 * **大小写有意义**:`m` 是毫,`M` 是兆。一个 case-insensitive 的解析器会把
 * 3 毫变成 3 兆 —— 正是这张表存在的理由。
 */
export const SI_PREFIX_EXP: Readonly<Record<string, number>> = {
  a: -18,
  f: -15,
  p: -12,
  n: -9,
  u: -6,
  "µ": -6, // U+00B5 MICRO SIGN
  "μ": -6, // U+03BC GREEK SMALL LETTER MU —— 多数键盘打出来的那个
  m: -3,
  k: 3,
  M: 6,
  G: 9,
};

/** 前缀 → 倍率。给 parity 测试用(Python 那边存的是倍率不是指数)。 */
export const SI_PREFIX_FACTORS: Readonly<Record<string, number>> =
  Object.fromEntries(
    Object.entries(SI_PREFIX_EXP).map(([k, e]) => [k, Number(`1e${e}`)]),
  );

const PREFIX_CHARS = Object.keys(SI_PREFIX_EXP).join("");

/**
 * **整串**匹配一个数值,可带一个 SI 前缀。
 *
 * 整串是关键:`parseFloat` 式的「能读多少读多少」会把 `1.5nA` 读成 `1.5`,
 * 而 `1.5` 是一个完美合法的安培数 —— 那正是这里要防的那次事故的形状。
 */
const NUM_RE = new RegExp(
  `^([+-]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:[eE][+-]?\\d+)?)\\s*([${PREFIX_CHARS}])?$`,
);

// ── 契约(镜像 `ParamSpecRow`,故意不含 default —— 见文件头 ①)────────────────

export interface ParamSpecLike {
  name: string;
  /** `float` / `int` / `str` / `bool`。 */
  type?: string | null;
  unit?: string | null;
  min_value?: number | null;
  max_value?: number | null;
  help?: string | null;
  choices?: unknown[] | null;
  /** `ParamSpec.default is None` ⇒ 必填。 */
  required?: boolean | null;
}

export type ParamValue = number | string | boolean;

export interface FieldParse {
  /** 还没填。**不是错误** —— 必填与否由 `buildCreateParams` 在提交那一刻判。 */
  empty: boolean;
  /** 要发上去的那个值。`null` = 没有可发的值。 */
  value: ParamValue | null;
  /** 一句人读的错误。空串 = 没问题。 */
  error: string;
  /** 系统**实际**收到的那个数,回读给人看。空串 = 没有可回读的。 */
  echo: string;
}

const EMPTY: FieldParse = { empty: true, value: null, error: "", echo: "" };

// ── 数值解析 ─────────────────────────────────────────────────────────────────

/** 把一个数渲染成人读的量(`1.50 nA`);无单位时就是这个数本身。 */
function qty(value: number, unit: string): string {
  return unit ? fmtSI(value, unit) : String(value);
}

/**
 * 回读:**人读的量 + 真正上线的那个数**,两个都写出来。
 *
 * 只写 `1.50 nA` 的话,量级错了看得见但对不上账;只写 `1.5e-9` 的话,用户要在
 * 针尖底下心算指数 —— 那正是 `units.ts` 当初被抠出来的那条现场反馈。两个都写,
 * 一眼能对上「我想要的」和「它收到的」。
 */
export function fieldEcho(value: number, unit: string): string {
  if (!Number.isFinite(value)) return "";
  if (!unit) return String(value);
  return `${fmtSI(value, unit)}（= ${String(value)} ${unit}）`;
}

/**
 * 一段文字 → 一个基本单位下的数。
 *
 * 接受:`1.5`、`-2`、`5e-3`、`1.5n`、`5nm`(末尾的单位会被剥掉)。
 * 不接受:任何整串匹配不上的东西 —— **绝不截一半**。
 */
export function parseQuantity(
  raw: string,
  unit: string,
): { value: number | null; error: string } {
  const s = String(raw ?? "").trim();
  if (!s) return { value: null, error: "" };

  // 末尾的单位逐字剥掉(区分大小写),这样 `5nm` / `1.5 nA` 都写得出来。
  let head = s;
  let strippedUnit = false;
  if (unit && s.length > unit.length && s.endsWith(unit)) {
    head = s.slice(0, s.length - unit.length).trim();
    strippedUnit = true;
  }

  // 单位本身就是一个前缀字母时会撞车:`5m` 在一个「米」的格子里,既可以读成
  // 5 米,也可以读成 5 毫米。差一千倍。**不猜** —— 猜错的那一半没有任何东西
  // 会说出来,而这正是「兜底值合理得让人看不出兜底发生了」那一类。
  if (strippedUnit && unit.length === 1 && unit in SI_PREFIX_EXP
      && /[\d.]$/.test(head)) {
    return {
      value: null,
      error:
        `「${s}」有歧义：末尾的 ${unit} 既是单位「${unit}」，也是 SI 前缀` +
        `（${unit} = 1e${SI_PREFIX_EXP[unit]}），两种读法差 1000 倍。` +
        `要 ${head} ${unit} 就写 ${head}；要 ${head} ${unit}${unit} 就写 ` +
        `${head}${unit}${unit} 或 ${head}e${SI_PREFIX_EXP[unit]}。`,
    };
  }

  const m = NUM_RE.exec(head);
  if (!m) {
    return {
      value: null,
      error:
        `「${s}」看不懂。写一个数就行（${unit ? `单位 ${unit}；` : ""}` +
        `可以带 SI 前缀，如 1.5n${unit}、5e-9）。` +
        `**这里不会「读到哪算哪」** —— 一个被读成 1.5 的 1.5n 是一次量级事故。`,
    };
  }
  const mant = m[1] ?? "";
  const pfx = m[2] ?? "";
  let value: number;
  if (!pfx) {
    value = Number(mant);
  } else if (!/[eE]/.test(mant)) {
    // 走字符串拼指数而不是乘法:`1.5 * 1e-9` 会引入一个二进制舍入尾巴,
    // 而回读框里一串 `1.5000000000000002e-9` 会被当成系统改了你的数。
    value = Number(`${mant}e${SI_PREFIX_EXP[pfx]}`);
  } else {
    value = Number(mant) * Number(`1e${SI_PREFIX_EXP[pfx]}`);
  }
  if (!Number.isFinite(value)) {
    return { value: null, error: `「${s}」不是一个有限的数` };
  }
  return { value, error: "" };
}

// ── 逐字段 ───────────────────────────────────────────────────────────────────

function choiceList(spec: ParamSpecLike): string[] {
  return (spec.choices ?? []).map((c) => String(c));
}

/**
 * 一个格子里那串字 → 要发上去的值(或者一句错误)。
 *
 * 判定顺序**逐条对着** `mast/conduct/validator.check_params`:
 * 类型 → choices(非空时 min/max 不参与) → 包络。顺序不同会让同一个输入在两边
 * 得到不同的那句错误,而用户会以为自己改的不是同一件事。
 */
export function parseParamInput(raw: string, spec: ParamSpecLike): FieldParse {
  const type = String(spec.type || "str");
  const unit = String(spec.unit || "");
  const s = String(raw ?? "").trim();
  if (!s) return EMPTY;

  if (type === "bool") {
    if (s === "true") return { empty: false, value: true, error: "", echo: "是" };
    if (s === "false") return { empty: false, value: false, error: "", echo: "否" };
    return { empty: false, value: null, error: `「${s}」不是 true/false`, echo: "" };
  }

  const choices = choiceList(spec);

  if (type === "str") {
    if (choices.length && !choices.includes(s)) {
      return {
        empty: false,
        value: null,
        error: `「${s}」不在允许集里：${choices.join(" / ")}`,
        echo: "",
      };
    }
    // 字符串参数**不做数值解析**:`bias_series_v` 是一串逗号分隔的偏压,
    // 把它当数字读会在第一个逗号处「读到哪算哪」。它的形状由后端的技能校验。
    return { empty: false, value: s, error: "", echo: "" };
  }

  const { value, error } = parseQuantity(s, unit);
  if (error) return { empty: false, value: null, error, echo: "" };
  if (value == null) return EMPTY;

  if (type === "int" && !Number.isInteger(value)) {
    return {
      empty: false,
      value: null,
      error: `${spec.name} 要一个整数，${value} 不是`,
      echo: "",
    };
  }

  if (choices.length) {
    // choices 非空时 min/max **不参与判定** —— 与 check_params 同一条。
    if (!choices.includes(String(value))) {
      return {
        empty: false,
        value: null,
        error: `${value} 不在允许集里：${choices.join(" / ")}`,
        echo: "",
      };
    }
    return { empty: false, value, error: "", echo: fieldEcho(value, unit) };
  }

  // ── 包络:**拒绝,不夹紧** ──────────────────────────────────────────────
  //
  // 夹到边界会把一次越界输入变成一次看起来完全正常的运行。比较用的是
  // `<` / `>`(等于边界是合法的),逐字对着 check_params —— 两边一个用 `<`
  // 一个用 `<=`,边界上那个值就会在这里过、在那边被拒。
  const lo = spec.min_value;
  const hi = spec.max_value;
  if (lo != null && value < lo) {
    return {
      empty: false,
      value,
      error:
        `${qty(value, unit)} 低于下限 ${qty(lo, unit)} —— 拒绝，不夹紧` +
        `（夹到边界的话，你以为设了 ${qty(value, unit)}，机器跑的是 ` +
        `${qty(lo, unit)}，两个数不会在任何日志里对上）`,
      echo: fieldEcho(value, unit),
    };
  }
  if (hi != null && value > hi) {
    return {
      empty: false,
      value,
      error:
        `${qty(value, unit)} 高于上限 ${qty(hi, unit)} —— 拒绝，不夹紧` +
        `（量级填错一位就是这个样子：1.5 nA 与 1.5 A 差十亿倍）`,
      echo: fieldEcho(value, unit),
    };
  }

  return { empty: false, value, error: "", echo: fieldEcho(value, unit) };
}

/** 包络那一行提示。没有包络就没有这句话(不编一个「无限制」出来)。 */
export function envelopeHint(spec: ParamSpecLike): string {
  const unit = String(spec.unit || "");
  const lo = spec.min_value;
  const hi = spec.max_value;
  const choices = choiceList(spec);
  if (choices.length) return `只能是：${choices.join(" / ")}`;
  if (lo != null && hi != null) {
    return `允许 ${qty(lo, unit)} – ${qty(hi, unit)}（超出即拒绝，不夹紧）`;
  }
  if (lo != null) return `不得低于 ${qty(lo, unit)}（超出即拒绝，不夹紧）`;
  if (hi != null) return `不得高于 ${qty(hi, unit)}（超出即拒绝，不夹紧）`;
  return "";
}

// ── 整张表 ───────────────────────────────────────────────────────────────────

export interface BuildResult {
  /** 要放进 `POST /api/conducts` 的 `params`。 */
  params: Record<string, ParamValue>;
  /** 字段名 → 一句错误(本地看出来的)。 */
  fieldErrors: Record<string, string>;
  /** 必填却空着的字段名。 */
  missing: string[];
  /** 本地没看出问题。**这不代表能建成** —— 真源在服务端。 */
  ok: boolean;
}

/**
 * 换模板时那张表的初值:**每个格子都是空的,没有例外。**
 *
 * 这个函数存在的唯一理由是让「一个物理量默认值都不预填」成为一条**测得出来**的
 * 规则,而不是一句写在注释里的自律。后端刻意不回 `default`,所以这里就算想预填
 * 也没有东西可填 —— 两道一起,一道是拿不到,一道是不去拿。
 */
export function initialValues(specs: ParamSpecLike[]): Record<string, string> {
  const out: Record<string, string> = {};
  for (const spec of specs ?? []) out[String(spec.name)] = "";
  return out;
}

/**
 * 整张表 → 请求体。
 *
 * ⚠️ 返回的 `ok=false` **不该被用来禁用提交按钮**。本地判断只是提前告知:一个
 * 本地判错就再也提交不上去的表单,是「能停不能解」的又一例,而这一侧的判断迟早
 * 会与 `check_params` 差一点(比如边界上的 `<` 与 `<=`)。所以按钮永远能按,
 * 能解析出来的都发上去,由后端说最后那句话。
 */
export function buildCreateParams(
  specs: ParamSpecLike[],
  raw: Record<string, string>,
): BuildResult {
  const params: Record<string, ParamValue> = {};
  const fieldErrors: Record<string, string> = {};
  const missing: string[] = [];
  for (const spec of specs ?? []) {
    const p = parseParamInput(raw?.[spec.name] ?? "", spec);
    if (p.empty) {
      // `required === false` 的字段留空 = 用模板的默认值。今天没有这样的参数
      // (`default` 全是 None),但契约允许,所以这里不假设。
      if (spec.required !== false) missing.push(spec.name);
      continue;
    }
    if (p.error) fieldErrors[spec.name] = p.error;
    // 有值就发上去 —— 包括超包络的那个。让**后端**说那句拒绝的话,这样本地
    // 判据万一比后端严,用户也不会被自己这一侧锁死。
    if (p.value != null) params[spec.name] = p.value;
  }
  return {
    params,
    fieldErrors,
    missing,
    ok: missing.length === 0 && Object.keys(fieldErrors).length === 0,
  };
}

/** 提交前那句提示。`ok` 时是空串 —— 没问题就别说话。 */
export function preSubmitNote(r: BuildResult): string {
  const bits: string[] = [];
  if (r.missing.length) bits.push(`${r.missing.length} 个必填格子还空着`);
  const bad = Object.keys(r.fieldErrors).length;
  if (bad) bits.push(`${bad} 个格子填的东西这一侧就看得出有问题`);
  if (!bits.length) return "";
  return `${bits.join("，")}。还是会提交 —— 最后那句话由后端说；红字见下面各格。`;
}

// ── 后端回来之后 ─────────────────────────────────────────────────────────────

export interface ParamEchoLike {
  name?: string | null;
  value?: unknown;
  unit?: string | null;
  ok?: boolean | null;
  error?: string | null;
}

export interface CreateOutcomeLike {
  ok?: boolean | null;
  conduct_id?: string | null;
  status?: string | null;
  params_echo?: ParamEchoLike[] | null;
  errors?: string[] | null;
  degraded?: boolean | null;
}

/**
 * findings 的字符串形状:`[code] where: message`(`validator.Finding.__str__`)。
 * `where` 可以带点(步 id),所以中间那段不能只吃 `\w`。
 */
const FINDING_RE = /^\[([a-z_]+)\]\s*([^:]*):\s*([\s\S]*)$/;

/**
 * 后端的逐字段回显 → 字段名 → 那句错误。
 *
 * 真源是 `params_echo`(后端按 `Finding.where` 分好组的),不是自己再去拆
 * `errors` 里那串字符串。
 */
export function serverFieldErrors(
  out: CreateOutcomeLike | null | undefined,
): Record<string, string> {
  const map: Record<string, string> = {};
  for (const row of out?.params_echo ?? []) {
    const name = String(row?.name ?? "");
    if (!name) continue;
    if (row?.ok === false) map[name] = String(row?.error || "后端拒绝了这个值");
  }
  return map;
}

/**
 * **没能落到任何一个格子上**的那些错误。
 *
 * 这个函数就是为了防一件事:把 `errors` 全当成逐字段的,于是像
 * `ActiveConductExists`(单活跃不变式)、`没有这个模板`、`conduct 层不可用`
 * 这种不带字段名的错误**一个字都不会出现在屏幕上** —— 用户按下建立,什么都
 * 没发生,也没有任何解释。逐字段回显是好东西,但它不能变成一个吞掉其余一切的
 * 漏斗。
 */
export function unassignedErrors(
  out: CreateOutcomeLike | null | undefined,
): string[] {
  const shown = new Set(Object.keys(serverFieldErrors(out)));
  return (out?.errors ?? [])
    .map((e) => String(e ?? ""))
    .filter((e) => e.length > 0)
    .filter((e) => {
      const m = FINDING_RE.exec(e);
      // 不是 finding 形状的(纯异常消息)⇒ 永远显示。
      if (!m) return true;
      const where = (m[2] ?? "").trim();
      // `where` 抽不出来时也显示:「我没认出这条属于谁」不是「这条不用显示」。
      return !where || !shown.has(where);
    });
}

/**
 * 非 2xx 的响应体 → 一个能显示的结果。
 *
 * `ConductCreateResponse` 只是**其中一种**回包:请求体本身不合法时 FastAPI 回的
 * 是 `{detail: [...]}`,里面一个 `errors` 都没有。照单当成 CreateOutcome 的话,
 * `unassignedErrors` 会得到空数组,屏幕上于是显示「后端没说为什么」—— 而后端明明
 * 说了,只是说在另一个字段里。「读不到」被折叠成了一个具体的值,又一次。
 */
export function normalizeCreateError(
  body: unknown,
  status: number,
): CreateOutcomeLike {
  const b = (body && typeof body === "object" ? body : {}) as Record<string, unknown>;
  if (Array.isArray(b.errors) || Array.isArray(b.params_echo)) {
    return b as CreateOutcomeLike;
  }
  const detail = b.detail;
  if (typeof detail === "string" && detail) {
    return { ok: false, errors: [`HTTP ${status}：${detail}`] };
  }
  if (Array.isArray(detail) && detail.length) {
    // pydantic 的逐条 422。`loc` 尾巴就是字段名,带上它,否则「哪个字段」要人猜。
    return {
      ok: false,
      errors: detail.map((d) => {
        const one = (d && typeof d === "object" ? d : {}) as Record<string, unknown>;
        const loc = Array.isArray(one.loc) ? one.loc.map(String).join(".") : "";
        const msg = String(one.msg ?? one.type ?? "请求体不合法");
        return `HTTP ${status}：${loc ? `${loc} ` : ""}${msg}`;
      }),
    };
  }
  return {
    ok: false,
    errors: [`HTTP ${status}：后端回了一个这一侧看不懂的响应体（不是「没有原因」）`],
  };
}

/** 提交结果那句 toast。 */
export function createOutcomeMessage(
  out: CreateOutcomeLike | null | undefined,
  ok: boolean,
): { text: string; tone: "ok" | "err" } {
  if (ok && out?.conduct_id) {
    return {
      text: `草稿建好了（${out.conduct_id}）。它还**没有**开始跑 —— 下一步是批准。`,
      tone: "ok",
    };
  }
  const first = unassignedErrors(out)[0] || "";
  const nField = Object.keys(serverFieldErrors(out)).length;
  if (nField) {
    return {
      text: `没建成：${nField} 个参数被拒绝${first ? `；${first}` : ""}（见下面各格）`,
      tone: "err",
    };
  }
  return { text: `没建成：${first || "后端没说为什么"}`, tone: "err" };
}

// ── 模板下拉 ─────────────────────────────────────────────────────────────────

export interface TemplateRowLike {
  spec_id?: string | null;
  spec_version?: number | null;
  title?: string | null;
  stages?: string[] | null;
  approvable?: boolean | null;
  findings?: string[] | null;
  checks_skipped?: string[] | null;
  params_schema?: ParamSpecLike[] | null;
}

export type TemplateListState =
  | { kind: "loading"; message: string }
  | { kind: "unreadable"; message: string }
  | { kind: "empty"; message: string }
  | { kind: "ready"; message: string };

/**
 * 模板列表现在是什么处境。
 *
 * **「读不到」与「没有」是两句话。** 拉不到时说「拉不到模板列表」并带上为什么;
 * 真的一个模板都没有时说的是另一句。把前者显示成一个空下拉框,用户会以为这台
 * 机器上就是没有模板 —— 一次故障被读成了一条事实,而那正是本仓记了一整页的
 * 「读不到被折叠成一个具体的值」。
 */
export function templateListState(q: {
  isPending?: boolean | null;
  isError?: boolean | null;
  error?: unknown;
  data?: { templates?: unknown[] | null; degraded?: boolean | null; reason?: string | null } | null;
}): TemplateListState {
  if (q?.isPending) return { kind: "loading", message: "正在取模板列表…" };
  if (q?.isError) {
    const msg = String((q.error as Error)?.message ?? q.error ?? "");
    return {
      kind: "unreadable",
      message: `拉不到模板列表${msg ? `：${msg}` : ""} —— 这是读不到，不是「没有模板」`,
    };
  }
  const d = q?.data;
  if (!d) {
    return { kind: "unreadable", message: "拉不到模板列表（没有响应体）—— 这是读不到，不是「没有模板」" };
  }
  if (d.degraded) {
    return {
      kind: "unreadable",
      message: `拉不到模板列表：${d.reason || "后端说它降级了，但没说为什么"}`,
    };
  }
  if (!(d.templates ?? []).length) {
    return {
      kind: "empty",
      message: "这台机器上一个 conduct 模板都没有（模板在 mast/conduct/templates/ 里注册）",
    };
  }
  return { kind: "ready", message: "" };
}

/** 下拉框里那一行的字。批不下去的模板要**当场看得出来**。 */
export function templateOptionLabel(row: TemplateRowLike): string {
  const id = String(row?.spec_id ?? "");
  const title = String(row?.title ?? "") || id;
  const ver = row?.spec_version ? ` v${row.spec_version}` : "";
  return row?.approvable === false ? `${title}${ver}（这台机器上批不下去）` : `${title}${ver}`;
}

/**
 * 选中的模板批不批得下去,以及**这不妨碍先建草稿**。
 *
 * 这个区分要紧:`POST /conduct` 只跑 `check_params`,批不下去的模板照样建得出
 * 草稿;真正被挡住的是 approve 那一步。把它说成「这个模板不能用」会让人去找一个
 * 错误的出路(换模板),而正确的出路是把缺的技能补上。
 */
export function templateApprovalNote(
  row: TemplateRowLike | null | undefined,
): { blocked: boolean; text: string } {
  if (!row || row.approvable !== false) return { blocked: false, text: "" };
  const skipped = (row.checks_skipped ?? []).length;
  return {
    blocked: true,
    text:
      "这个模板**现在批不下去**：草稿建得出来，但下一步的「批准」会被挡住。" +
      (skipped
        ? "注意其中有**根本没跑的检查**（下面「没能跑的检查」那一栏）——「没检查」不等于「检查通过」。"
        : "缺什么见下面。"),
  };
}

/** 选中的那个模板。选中项不在列表里就返回 `null`(不悄悄换一个顶上)。 */
export function pickTemplate(
  rows: TemplateRowLike[] | null | undefined,
  specId: string,
): TemplateRowLike | null {
  if (!specId) return null;
  return (rows ?? []).find((r) => String(r?.spec_id ?? "") === specId) ?? null;
}

// ── 实验归属 ─────────────────────────────────────────────────────────────────

export type ExperimentPickState =
  | { kind: "list"; message: string }
  | { kind: "manual"; message: string };

/**
 * 实验下拉能不能用。**用不了的时候要留一条手填的路** —— `experiment_id` 是必填
 * 的(没有实验归属的 conduct,产物没有落点),下拉拉不到就把新建整个堵死了,
 * 而那是「能停不能解」。
 */
export function experimentPickState(q: {
  isPending?: boolean | null;
  isError?: boolean | null;
  data?: { experiments?: unknown[] | null; degraded?: boolean | null } | null;
}): ExperimentPickState {
  if (q?.isPending) return { kind: "manual", message: "正在取实验列表…（也可以直接手填 ID）" };
  if (q?.isError || !q?.data || q.data.degraded) {
    return {
      kind: "manual",
      message: "读不到实验列表（不是「没有实验」）—— 手填一个实验 ID 也能建",
    };
  }
  if (!(q.data.experiments ?? []).length) {
    return { kind: "manual", message: "还没有实验 —— 手填一个 ID，或先去实验记录里建一个" };
  }
  return { kind: "list", message: "" };
}
