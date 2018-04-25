function add_sponsor(e, t, n, i, r, o) {
	var a = '<div style="float:left;margin:0 0 -1px -1px;border:solid 1px #ddd;' + ("width:" + t + "px;height:" + n + "px;") + '">';
	4 === arguments.length ? a += i: (a = (a = a + '<a target="_blank" href="' + o + '">') + '<img src="' + r + '">', a += "</a>"),
	a += "</div>",
	$(e).append(a)
}
function deleteTopic(e) {
	confirm("Delete this topic?") && postJSON("/api/topics/" + e + "/delete",
	function(e, t) {
		e ? alert(e.message || e) : location.assign("/discuss")
	})
}
function deleteReply(e) {
	confirm("Delete this reply?") && postJSON("/api/replies/" + e + "/delete",
	function(e, t) {
		e ? alert(e.message || e) : refresh()
	})
}
function getCookie(e) {
	var t = document.cookie.match("(^|;) ?" + e + "=([^;]*)(;|$)");
	return t ? t[2] : null
}
function setCookie(e, t, n) {
	var i = new Date((new Date).getTime() + 1e3 * n);
	document.cookie = e + "=" + t + ";path=/;expires=" + i.toGMTString() + ("https" === location.protocol ? ";secure": "")
}
function deleteCookie(e) {
	var t = new Date(0);
	document.cookie = e + "=deleted;path=/;expires=" + t.toGMTString() + ("https" === location.protocol ? ";secure": "")
}
function message(e, t, n, i) {
	0 == $("#modal-message").length && $("body").append('<div id="modal-message" class="uk-modal"><div class="uk-modal-dialog"><button type="button" class="uk-modal-close uk-close"></button><div class="uk-modal-header"></div><p class="x-message">msg</p></div></div>');
	var r = $("#modal-message");
	r.find("div.uk-modal-header").text(e || "Message"),
	n ? r.find("p.x-message").html(t || "") : r.find("p.x-message").text(t || ""),
	UIkit.modal("#modal-message").show()
}
function _get_code(e) {
	var t = $("#pre-" + e),
	n = $("#post-" + e),
	i = $("#textarea-" + e);
	return t.text() + i.val() + "\n" + (0 === n.length ? "": n.text())
}
function run_javascript(tid, btn) {
	var code = _get_code(tid); !
	function() {
		var buffer = "",
		_log = function(e) {
			console.log(e),
			buffer = buffer + e + "\n"
		},
		_warn = function(e) {
			console.warn(e),
			buffer = buffer + e + "\n"
		},
		_error = function(e) {
			console.error(e),
			buffer = buffer + e + "\n"
		},
		_console = {
			trace: _log,
			debug: _log,
			log: _log,
			info: _log,
			warn: _warn,
			error: _error
		};
		try {
			eval("(function() {\n var console = _console; \n" + code + "\n})();"),
			buffer && showCodeResult(btn, buffer)
		} catch(e) {
			showCodeError(btn, String(e))
		}
	} ()
}
function run_html(e, t) {
	var n = _get_code(e); !
	function() {
		var e = window.open("about:blank", "Online Practice", "width=640,height=480,resizeable=1,scrollbars=1");
		e.document.write(n),
		e.document.close()
	} ()
}
function _showCodeResult(e, t, n, i) {
	var r = $(e).next("div.x-code-result");
	if (void 0 === r.get(0) && ($(e).after('<div class="x-code-result x-code uk-alert"></div>'), r = $(e).next("div.x-code-result")), r.removeClass("uk-alert-danger"), i && r.addClass("uk-alert-danger"), n) r.html(t);
	else {
		var o = t.split("\n"),
		a = _.map(o,
		function(e) {
			return encodeHtml(e).replace(/ /g, "&nbsp;")
		}).join("<br>");
		r.html(a)
	}
}
function _history(s){
	var xmlHttp;
	//判断浏览器是否支持ActiveX控件
	exports(s);
	console.log('this is history',s);
}
function showCodeResult(e, t, n) {
	_showCodeResult(e, t, n)
}
function showCodeError(e, t, n) {
	_showCodeResult(e, t, n, !0)
}
function run_sql(e, t) {
	if (void 0 !== typeof alasql) {
		var n = _get_code(e),
		i = function(e) {
			if (0 === e.length) return "Empty result set";
			var t = _.keys(e[0]),
			n = _.map(e,
			function(e) {
				return _.map(t,
				function(t) {
					return e[t]
				})
			});
			return '<table class="uk-table"><thead><tr>' + $.map(t,
			function(e) {
				var t = e.indexOf("!");
				return t > 1 && (e = e.substring(t + 1)),
				"<th>" + encodeHtml(e) + "</th>"
			}).join("") + "</tr></thead><tbody>" + $.map(n,
			function(e) {
				return "<tr>" + $.map(e,
				function(e) {
					return void 0 === e && (e = "NULL"),
					"<td>" + encodeHtml(e) + "</td>"
				}).join("") + "</tr>"
			}).join("") + "</tbody></table>"
		}; !
		function() {
			var e, r, o = "",
			a = n.split("\n");
			for (a = _.map(a,
			function(e) {
				var t = e.indexOf("--");
				return t >= 0 && (e = e.substring(0, t)),
				e
			}), a = _.filter(a,
			function(e) {
				return "" !== e.trim()
			}), e = 0; e < a.length; e++) o = o + a[e] + "\n";
			for (a = _.filter(o.trim().split(";"),
			function(e) {
				return "" !== e.trim()
			}), r = null, error = null, e = 0; e < a.length; e++) {
				o = a[e];
				try {
					r = alasql(o)
				} catch(e) {
					error = e;
					break
				}
			}
			error ? showCodeError(t, "ERROR when execute SQL: " + o + "\n" + String(error)) : Array.isArray(r) ? showCodeResult(t, i(r), !0) : showCodeResult(t, r || "(empty)")
		} ()
	} else showCodeError(t, "错误：JavaScript嵌入式SQL引擎尚未加载完成，请稍后再试或者刷新页面！")
}
function run_python(e, t) {
	var n = $("#pre-" + e),
	i = $("#post-" + e),
	r = $("#textarea-" + e),
	o = $(t),
	a = o.find("i"),
	s = n.text() + r.val() + "\n" + (0 === i.length ? "": i.text());
	o.attr("disabled", "disabled"),
	a.addClass("uk-icon-spinner"),
	a.addClass("uk-icon-spin"),
	$.post("https://local.liaoxuefeng.com:39093/run", $.param({
		code: s
	})).done(function(e) {
		showCodeResult(t, e.output);
		_history(r.val());
	}).fail(function(e) {
		showCodeError(t, '<p>无法连接到Python代码运行助手。请在本机的存放learning.py的目录下运行命令：F:\\tool\\py\\py_tool python learning.py 如果看到Ready for Python code on port 39093...表示运行成功，不要关闭命令行窗口，最小化放到后台运行即可。</p>', !0)
	}).always(function() {
		a.removeClass("uk-icon-spinner"),
		a.removeClass("uk-icon-spin"),
		o.removeAttr("disabled")
	})
}
function adjustTextareaHeight(e) {
	var t = $(e),
	n = t.val().split("\n").length;
	n < 9 && (n = 9),
	t.attr("rows", "" + (n + 1))
}
function initCommentArea(e, t, n) {
	$("#x-comment-area").html($("#tplCommentArea").html());
	var i = $("#comment-make-button"),
	r = $("#comment-form"),
	o = (r.find("button[type=submit]"), r.find("button.x-cancel"));
	i.click(function() {
		r.showFormError(),
		r.show(),
		r.find("div.x-textarea").html("<textarea></textarea>");
		UIkit.htmleditor(r.find("textarea").get(0), {
			mode: "split",
			maxsplitsize: 600,
			markdown: !0
		});
		i.hide()
	}),
	o.click(function() {
		r.find("div.x-textarea").html(""),
		r.hide(),
		i.show()
	}),
	r.submit(function(i) {
		i.preventDefault(),
		r.postJSON("/api/comments/" + e + "/" + t, {
			tag: n,
			name: r.find("input[name=name]").val(),
			content: r.find("textarea").val()
		},
		function(e, t) {
			e || refresh("#comments")
		})
	})
}
function showSignin(e) {
	if (1 === g_signins.length && !e) return authFrom(g_signins[0].id);
	null === signinModal && (signinModal = UIkit.modal("#modal-signin", {
		bgclose: !1,
		center: !0
	})),
	signinModal.show()
}
function Template(e) {
	for (var t, n, i = ["var r=[];\nvar _html = function (str) { return str.replace(/&/g, '&amp;').replace(/\"/g, '&quot;').replace(/'/g, '&#39;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); };"], r = /\{\s*([a-zA-Z\.\_0-9()]+)(\s*\|\s*safe)?\s*\}/m, o = function(e) {
		i.push("r.push('" + e.replace(/\'/g, "\\'").replace(/\n/g, "\\n").replace(/\r/g, "\\r") + "');")
	}; n = r.exec(e);) n.index > 0 && o(e.slice(0, n.index)),
	n[2] ? i.push("r.push(String(this." + n[1] + "));") : i.push("r.push(_html(String(this." + n[1] + ")));"),
	e = e.substring(n.index + n[0].length);
	o(e),
	i.push("return r.join('');"),
	t = new Function(i.join("\n")),
	this.render = function(e) {
		return t.apply(e)
	}
}
function buildComments(e) {
	if (null === tplComment && (tplComment = new Template($("#tplComment").html())), null === tplCommentReply && (tplCommentReply = new Template($("#tplCommentReply").html())), null === tplCommentInfo && (tplCommentInfo = new Template($("#tplCommentInfo").html())), 0 === e.topics.length) return "<p>No comment yet.</p>";
	var t, n, i, r, o = [];
	e.page;
	for (t = 0; t < e.topics.length; t++) {
		if (i = e.topics[t], o.push("<li>"), o.push(tplComment.render(i)), o.push("<ul>"), i.replies.length > 0) for (n = 0; n < i.replies.length; n++) r = i.replies[n],
		o.push("<li>"),
		o.push(tplCommentReply.render(r)),
		o.push("</li>");
		o.push(tplCommentInfo.render(i)),
		o.push("</ul>"),
		o.push("</li>")
	}
	return o.join("")
}
function ajaxLoadComments(e, t, n) {
	var i = 'Error when loading. <a href="#0" onclick="ajaxLoadComments(\'' + e + "', '" + t + "', " + n + ')">Retry</a>';
	$insertInto = $("#" + e),
	$insertInto.html('<i class="uk-icon-spinner uk-icon-spin"></i> Loading...'),
	$.getJSON("/api/ref/" + t + "/topics?page=" + n).done(function(e) {
		e.error ? $insertInto.html(i) : ($insertInto.html(buildComments(e)), $insertInto.find(".x-auto-content").each(function() {
			makeCollapsable(this, 400)
		}))
	}).fail(function() {
		$insertInto.html(i)
	})
}
function makeCollapsable(e, t) {
	var n = $(e);
	if (n.height() <= t + 60) n.show();
	else {
		var i = t + "px";
		n.css("max-height", i),
		n.css("overflow", "hidden"),
		n.after('<p style="padding-left: 75px"><a href="#0"><i class="uk-icon-chevron-down"></i> Read More</a><a href="#0" style="display:none"><i class="uk-icon-chevron-up"></i> Collapse</a></p>');
		var r = "COLLAPSE-" + tmp_collapse;
		tmp_collapse++,
		n.parent().before('<div class="x-anchor"><a name="' + r + '"></a></div>');
		var o = n.next(),
		a = o.find("a:first"),
		s = o.find("a:last");
		a.click(function() {
			n.css("max-height", "none"),
			a.hide(),
			s.show()
		}),
		s.click(function() {
			n.css("max-height", i),
			s.hide(),
			a.show(),
			location.assign("#" + r)
		}),
		n.show()
	}
}
function loadComments(e) {
	$(function() {
		var t = !1,
		n = $(window),
		i = $("#x-comment-list").get(0).offsetTop,
		r = function() { ! t && window.pageYOffset + window.innerHeight >= i && (t = !0, n.off("scroll", r), ajaxLoadComments("x-comment-list", e, 1))
		};
		n.scroll(r),
		r()
	})
}
function encodeHtml(e) {
	return String(e).replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/'/g, "&#39;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
}
function toSmartDate(e) {
	if ("string" == typeof e && (e = parseInt(e)), isNaN(e)) return "";
	var t = new Date(g_time),
	n = "1分钟前",
	i = t.getTime() - e;
	if (i > 6048e5) {
		var r = new Date(e),
		o = r.getFullYear(),
		a = r.getMonth() + 1,
		s = r.getDate(),
		l = r.getHours(),
		c = r.getMinutes();
		n = (n = o === t.getFullYear() ? "": o + "-") + a + "-" + s + " " + l + ":" + (c < 10 ? "0": "") + c
	} else i >= 864e5 ? n = Math.floor(i / 864e5) + "天前": i >= 36e5 ? n = Math.floor(i / 36e5) + "小时前": i >= 6e4 && (n = Math.floor(i / 6e4) + "分钟前");
	return n
}
function parseQueryString() {
	var e, t, n, i, r = location.search,
	o = {};
	if (r && "?" === r.charAt(0)) for (i = r.substring(1).split("&"), e = 0; e < i.length; e++)(t = (n = i[e]).indexOf("=")) <= 0 || (o[n.substring(0, t)] = decodeURIComponent(n.substring(t + 1)).replace(/\+/g, " "));
	return o
}
function gotoPage(e) {
	var t = parseQueryString();
	t.page = e,
	location.assign("?" + $.param(t))
}
function refresh(e) {
	var t = parseQueryString();
	t.t = (new Date).getTime(),
	location.assign("?" + $.param(t) + (e || ""))
}
function _httpJSON(e, t, n, i) {
	var r = {
		type: e,
		dataType: "json"
	};
	"GET" === e && (r.url = t + "?" + n),
	"POST" === e && (r.url = t, r.data = JSON.stringify(n || {}), r.contentType = "application/json"),
	$.ajax(r).done(function(e) {
		return e && e.error ? i(e) : i(null, e)
	}).fail(function(e, t) {
		return i({
			error: "http_bad_response",
			data: "" + e.status,
			message: "网络好像出问题了 (HTTP " + e.status + ")"
		})
	})
}
function getJSON(e, t, n) {
	if (2 === arguments.length && (n = t, t = {}), "object" == typeof t) {
		var i = [];
		$.each(t,
		function(e, t) {
			i.push(e + "=" + encodeURIComponent(t))
		}),
		t = i.join("&")
	}
	_httpJSON("GET", e, t, n)
}
function postJSON(e, t, n) {
	2 === arguments.length && (n = t, t = {}),
	_httpJSON("POST", e, t, n)
}
function onAuthCallback(e, t) {
	null !== signinModal && signinModal.hide(),
	e || (g_user = {
		id: t.id,
		name: t.name,
		image_url: t.image_url
	},
	$(".x-user-name").text(g_user.name), $("#x-doc-style").text(".x-display-if-signin {}\n.x-display-if-not-signin { display: none; }\n"), "undefined" != typeof g_reload_after_signin && !0 === g_reload_after_signin ? location.reload() : "function" == typeof onAuthSuccess && onAuthSuccess())
}
function authFrom(e) {
	var t = "/auth/from/" + e,
	n = location.hostname.replace(/\./g, "_");
	if (isDesktop) window.open(t + "?jscallback=onAuthCallback", n, "top=200,left=400,width=600,height=380,directories=no,menubar=no,toolbar=no,resizable=no");
	else location.assign(t)
} !
function(e, t) {
	"object" == typeof module && "object" == typeof module.exports ? module.exports = e.document ? t(e, !0) : function(e) {
		if (!e.document) throw new Error("jQuery requires a window with a document");
		return t(e)
	}: t(e)
} ("undefined" != typeof window ? window: this,
function(e, t) {
	function n(e) {
		var t = "length" in e && e.length,
		n = Q.type(e);
		return "function" !== n && !Q.isWindow(e) && (!(1 !== e.nodeType || !t) || ("array" === n || 0 === t || "number" == typeof t && t > 0 && t - 1 in e))
	}
	function i(e, t, n) {
		if (Q.isFunction(t)) return Q.grep(e,
		function(e, i) {
			return !! t.call(e, i, e) !== n
		});
		if (t.nodeType) return Q.grep(e,
		function(e) {
			return e === t !== n
		});
		if ("string" == typeof t) {
			if (ae.test(t)) return Q.filter(t, e, n);
			t = Q.filter(t, e)
		}
		return Q.grep(e,
		function(e) {
			return U.call(t, e) >= 0 !== n
		})
	}
	function r(e, t) {
		for (; (e = e[t]) && 1 !== e.nodeType;);
		return e
	}
	function o(e) {
		var t = he[e] = {};
		return Q.each(e.match(de) || [],
		function(e, n) {
			t[n] = !0
		}),
		t
	}
	function a() {
		Z.removeEventListener("DOMContentLoaded", a, !1),
		e.removeEventListener("load", a, !1),
		Q.ready()
	}
	function s() {
		Object.defineProperty(this.cache = {},
		0, {
			get: function() {
				return {}
			}
		}),
		this.expando = Q.expando + s.uid++
	}
	function l(e, t, n) {
		var i;
		if (void 0 === n && 1 === e.nodeType) if (i = "data-" + t.replace(ye, "-$1").toLowerCase(), "string" == typeof(n = e.getAttribute(i))) {
			try {
				n = "true" === n || "false" !== n && ("null" === n ? null: +n + "" === n ? +n: ve.test(n) ? Q.parseJSON(n) : n)
			} catch(e) {}
			ge.set(e, t, n)
		} else n = void 0;
		return n
	}
	function c() {
		return ! 0
	}
	function u() {
		return ! 1
	}
	function d() {
		try {
			return Z.activeElement
		} catch(e) {}
	}
	function h(e, t) {
		return Q.nodeName(e, "table") && Q.nodeName(11 !== t.nodeType ? t: t.firstChild, "tr") ? e.getElementsByTagName("tbody")[0] || e.appendChild(e.ownerDocument.createElement("tbody")) : e
	}
	function f(e) {
		return e.type = (null !== e.getAttribute("type")) + "/" + e.type,
		e
	}
	function p(e) {
		var t = $e.exec(e.type);
		return t ? e.type = t[1] : e.removeAttribute("type"),
		e
	}
	function m(e, t) {
		for (var n = 0,
		i = e.length; i > n; n++) me.set(e[n], "globalEval", !t || me.get(t[n], "globalEval"))
	}
	function g(e, t) {
		var n, i, r, o, a, s, l, c;
		if (1 === t.nodeType) {
			if (me.hasData(e) && (o = me.access(e), a = me.set(t, o), c = o.events)) {
				delete a.handle,
				a.events = {};
				for (r in c) for (n = 0, i = c[r].length; i > n; n++) Q.event.add(t, r, c[r][n])
			}
			ge.hasData(e) && (s = ge.access(e), l = Q.extend({},
			s), ge.set(t, l))
		}
	}
	function v(e, t) {
		var n = e.getElementsByTagName ? e.getElementsByTagName(t || "*") : e.querySelectorAll ? e.querySelectorAll(t || "*") : [];
		return void 0 === t || t && Q.nodeName(e, t) ? Q.merge([e], n) : n
	}
	function y(e, t) {
		var n = t.nodeName.toLowerCase();
		"input" === n && xe.test(e.type) ? t.checked = e.checked: ("input" === n || "textarea" === n) && (t.defaultValue = e.defaultValue)
	}
	function b(t, n) {
		var i, r = Q(n.createElement(t)).appendTo(n.body),
		o = e.getDefaultComputedStyle && (i = e.getDefaultComputedStyle(r[0])) ? i.display: Q.css(r[0], "display");
		return r.detach(),
		o
	}
	function w(e) {
		var t = Z,
		n = Ie[e];
		return n || ("none" !== (n = b(e, t)) && n || (Pe = (Pe || Q("<iframe frameborder='0' width='0' height='0'/>")).appendTo(t.documentElement), (t = Pe[0].contentDocument).write(), t.close(), n = b(e, t), Pe.detach()), Ie[e] = n),
		n
	}
	function k(e, t, n) {
		var i, r, o, a, s = e.style;
		return (n = n || He(e)) && (a = n.getPropertyValue(t) || n[t]),
		n && ("" !== a || Q.contains(e.ownerDocument, e) || (a = Q.style(e, t)), We.test(a) && ze.test(t) && (i = s.width, r = s.minWidth, o = s.maxWidth, s.minWidth = s.maxWidth = s.width = a, a = n.width, s.width = i, s.minWidth = r, s.maxWidth = o)),
		void 0 !== a ? a + "": a
	}
	function x(e, t) {
		return {
			get: function() {
				return e() ? void delete this.get: (this.get = t).apply(this, arguments)
			}
		}
	}
	function _(e, t) {
		if (t in e) return t;
		for (var n = t[0].toUpperCase() + t.slice(1), i = t, r = Ge.length; r--;) if ((t = Ge[r] + n) in e) return t;
		return i
	}
	function C(e, t, n) {
		var i = Ye.exec(t);
		return i ? Math.max(0, i[1] - (n || 0)) + (i[2] || "px") : t
	}
	function S(e, t, n, i, r) {
		for (var o = n === (i ? "border": "content") ? 4 : "width" === t ? 1 : 0, a = 0; 4 > o; o += 2)"margin" === n && (a += Q.css(e, n + we[o], !0, r)),
		i ? ("content" === n && (a -= Q.css(e, "padding" + we[o], !0, r)), "margin" !== n && (a -= Q.css(e, "border" + we[o] + "Width", !0, r))) : (a += Q.css(e, "padding" + we[o], !0, r), "padding" !== n && (a += Q.css(e, "border" + we[o] + "Width", !0, r)));
		return a
	}
	function M(e, t, n) {
		var i = !0,
		r = "width" === t ? e.offsetWidth: e.offsetHeight,
		o = He(e),
		a = "border-box" === Q.css(e, "boxSizing", !1, o);
		if (0 >= r || null == r) {
			if ((0 > (r = k(e, t, o)) || null == r) && (r = e.style[t]), We.test(r)) return r;
			i = a && (K.boxSizingReliable() || r === e.style[t]),
			r = parseFloat(r) || 0
		}
		return r + S(e, t, n || (a ? "border": "content"), i, o) + "px"
	}
	function T(e, t) {
		for (var n, i, r, o = [], a = 0, s = e.length; s > a; a++)(i = e[a]).style && (o[a] = me.get(i, "olddisplay"), n = i.style.display, t ? (o[a] || "none" !== n || (i.style.display = ""), "" === i.style.display && ke(i) && (o[a] = me.access(i, "olddisplay", w(i.nodeName)))) : (r = ke(i), "none" === n && r || me.set(i, "olddisplay", r ? n: Q.css(i, "display"))));
		for (a = 0; s > a; a++)(i = e[a]).style && (t && "none" !== i.style.display && "" !== i.style.display || (i.style.display = t ? o[a] || "": "none"));
		return e
	}
	function D(e, t, n, i, r) {
		return new D.prototype.init(e, t, n, i, r)
	}
	function L() {
		return setTimeout(function() {
			Ve = void 0
		}),
		Ve = Q.now()
	}
	function O(e, t) {
		var n, i = 0,
		r = {
			height: e
		};
		for (t = t ? 1 : 0; 4 > i; i += 2 - t) n = we[i],
		r["margin" + n] = r["padding" + n] = e;
		return t && (r.opacity = r.width = e),
		r
	}
	function N(e, t, n) {
		for (var i, r = (et[t] || []).concat(et["*"]), o = 0, a = r.length; a > o; o++) if (i = r[o].call(n, t, e)) return i
	}
	function A(e, t) {
		var n, i, r, o, a;
		for (n in e) if (i = Q.camelCase(n), r = t[i], o = e[n], Q.isArray(o) && (r = o[1], o = e[n] = o[0]), n !== i && (e[i] = o, delete e[n]), (a = Q.cssHooks[i]) && "expand" in a) {
			o = a.expand(o),
			delete e[i];
			for (n in o) n in e || (e[n] = o[n], t[n] = r)
		} else t[i] = r
	}
	function E(e, t, n) {
		var i, r, o = 0,
		a = Je.length,
		s = Q.Deferred().always(function() {
			delete l.elem
		}),
		l = function() {
			if (r) return ! 1;
			for (var t = Ve || L(), n = Math.max(0, c.startTime + c.duration - t), i = 1 - (n / c.duration || 0), o = 0, a = c.tweens.length; a > o; o++) c.tweens[o].run(i);
			return s.notifyWith(e, [c, i, n]),
			1 > i && a ? n: (s.resolveWith(e, [c]), !1)
		},
		c = s.promise({
			elem: e,
			props: Q.extend({},
			t),
			opts: Q.extend(!0, {
				specialEasing: {}
			},
			n),
			originalProperties: t,
			originalOptions: n,
			startTime: Ve || L(),
			duration: n.duration,
			tweens: [],
			createTween: function(t, n) {
				var i = Q.Tween(e, c.opts, t, n, c.opts.specialEasing[t] || c.opts.easing);
				return c.tweens.push(i),
				i
			},
			stop: function(t) {
				var n = 0,
				i = t ? c.tweens.length: 0;
				if (r) return this;
				for (r = !0; i > n; n++) c.tweens[n].run(1);
				return t ? s.resolveWith(e, [c, t]) : s.rejectWith(e, [c, t]),
				this
			}
		}),
		u = c.props;
		for (A(u, c.opts.specialEasing); a > o; o++) if (i = Je[o].call(c, e, u, c.opts)) return i;
		return Q.map(u, N, c),
		Q.isFunction(c.opts.start) && c.opts.start.call(e, c),
		Q.fx.timer(Q.extend(l, {
			elem: e,
			anim: c,
			queue: c.opts.queue
		})),
		c.progress(c.opts.progress).done(c.opts.done, c.opts.complete).fail(c.opts.fail).always(c.opts.always)
	}
	function $(e) {
		return function(t, n) {
			"string" != typeof t && (n = t, t = "*");
			var i, r = 0,
			o = t.toLowerCase().match(de) || [];
			if (Q.isFunction(n)) for (; i = o[r++];)"+" === i[0] ? (i = i.slice(1) || "*", (e[i] = e[i] || []).unshift(n)) : (e[i] = e[i] || []).push(n)
		}
	}
	function q(e, t, n, i) {
		function r(s) {
			var l;
			return o[s] = !0,
			Q.each(e[s] || [],
			function(e, s) {
				var c = s(t, n, i);
				return "string" != typeof c || a || o[c] ? a ? !(l = c) : void 0 : (t.dataTypes.unshift(c), r(c), !1)
			}),
			l
		}
		var o = {},
		a = e === gt;
		return r(t.dataTypes[0]) || !o["*"] && r("*")
	}
	function j(e, t) {
		var n, i, r = Q.ajaxSettings.flatOptions || {};
		for (n in t) void 0 !== t[n] && ((r[n] ? e: i || (i = {}))[n] = t[n]);
		return i && Q.extend(!0, e, i),
		e
	}
	function P(e, t, n) {
		for (var i, r, o, a, s = e.contents,
		l = e.dataTypes;
		"*" === l[0];) l.shift(),
		void 0 === i && (i = e.mimeType || t.getResponseHeader("Content-Type"));
		if (i) for (r in s) if (s[r] && s[r].test(i)) {
			l.unshift(r);
			break
		}
		if (l[0] in n) o = l[0];
		else {
			for (r in n) {
				if (!l[0] || e.converters[r + " " + l[0]]) {
					o = r;
					break
				}
				a || (a = r)
			}
			o = o || a
		}
		return o ? (o !== l[0] && l.unshift(o), n[o]) : void 0
	}
	function I(e, t, n, i) {
		var r, o, a, s, l, c = {},
		u = e.dataTypes.slice();
		if (u[1]) for (a in e.converters) c[a.toLowerCase()] = e.converters[a];
		for (o = u.shift(); o;) if (e.responseFields[o] && (n[e.responseFields[o]] = t), !l && i && e.dataFilter && (t = e.dataFilter(t, e.dataType)), l = o, o = u.shift()) if ("*" === o) o = l;
		else if ("*" !== l && l !== o) {
			if (! (a = c[l + " " + o] || c["* " + o])) for (r in c) if ((s = r.split(" "))[1] === o && (a = c[l + " " + s[0]] || c["* " + s[0]])) { ! 0 === a ? a = c[r] : !0 !== c[r] && (o = s[0], u.unshift(s[1]));
				break
			}
			if (!0 !== a) if (a && e.throws) t = a(t);
			else try {
				t = a(t)
			} catch(e) {
				return {
					state: "parsererror",
					error: a ? e: "No conversion from " + l + " to " + o
				}
			}
		}
		return {
			state: "success",
			data: t
		}
	}
	function z(e, t, n, i) {
		var r;
		if (Q.isArray(t)) Q.each(t,
		function(t, r) {
			n || kt.test(e) ? i(e, r) : z(e + "[" + ("object" == typeof r ? t: "") + "]", r, n, i)
		});
		else if (n || "object" !== Q.type(t)) i(e, t);
		else for (r in t) z(e + "[" + r + "]", t[r], n, i)
	}
	function W(e) {
		return Q.isWindow(e) ? e: 9 === e.nodeType && e.defaultView
	}
	var H = [],
	F = H.slice,
	Y = H.concat,
	R = H.push,
	U = H.indexOf,
	B = {},
	G = B.toString,
	V = B.hasOwnProperty,
	K = {},
	Z = e.document,
	X = "2.1.4",
	Q = function(e, t) {
		return new Q.fn.init(e, t)
	},
	J = /^[\s\uFEFF\xA0]+|[\s\uFEFF\xA0]+$/g,
	ee = /^-ms-/,
	te = /-([\da-z])/gi,
	ne = function(e, t) {
		return t.toUpperCase()
	};
	Q.fn = Q.prototype = {
		jquery: X,
		constructor: Q,
		selector: "",
		length: 0,
		toArray: function() {
			return F.call(this)
		},
		get: function(e) {
			return null != e ? 0 > e ? this[e + this.length] : this[e] : F.call(this)
		},
		pushStack: function(e) {
			var t = Q.merge(this.constructor(), e);
			return t.prevObject = this,
			t.context = this.context,
			t
		},
		each: function(e, t) {
			return Q.each(this, e, t)
		},
		map: function(e) {
			return this.pushStack(Q.map(this,
			function(t, n) {
				return e.call(t, n, t)
			}))
		},
		slice: function() {
			return this.pushStack(F.apply(this, arguments))
		},
		first: function() {
			return this.eq(0)
		},
		last: function() {
			return this.eq( - 1)
		},
		eq: function(e) {
			var t = this.length,
			n = +e + (0 > e ? t: 0);
			return this.pushStack(n >= 0 && t > n ? [this[n]] : [])
		},
		end: function() {
			return this.prevObject || this.constructor(null)
		},
		push: R,
		sort: H.sort,
		splice: H.splice
	},
	Q.extend = Q.fn.extend = function() {
		var e, t, n, i, r, o, a = arguments[0] || {},
		s = 1,
		l = arguments.length,
		c = !1;
		for ("boolean" == typeof a && (c = a, a = arguments[s] || {},
		s++), "object" == typeof a || Q.isFunction(a) || (a = {}), s === l && (a = this, s--); l > s; s++) if (null != (e = arguments[s])) for (t in e) n = a[t],
		i = e[t],
		a !== i && (c && i && (Q.isPlainObject(i) || (r = Q.isArray(i))) ? (r ? (r = !1, o = n && Q.isArray(n) ? n: []) : o = n && Q.isPlainObject(n) ? n: {},
		a[t] = Q.extend(c, o, i)) : void 0 !== i && (a[t] = i));
		return a
	},
	Q.extend({
		expando: "jQuery" + (X + Math.random()).replace(/\D/g, ""),
		isReady: !0,
		error: function(e) {
			throw new Error(e)
		},
		noop: function() {},
		isFunction: function(e) {
			return "function" === Q.type(e)
		},
		isArray: Array.isArray,
		isWindow: function(e) {
			return null != e && e === e.window
		},
		isNumeric: function(e) {
			return ! Q.isArray(e) && e - parseFloat(e) + 1 >= 0
		},
		isPlainObject: function(e) {
			return "object" === Q.type(e) && !e.nodeType && !Q.isWindow(e) && !(e.constructor && !V.call(e.constructor.prototype, "isPrototypeOf"))
		},
		isEmptyObject: function(e) {
			var t;
			for (t in e) return ! 1;
			return ! 0
		},
		type: function(e) {
			return null == e ? e + "": "object" == typeof e || "function" == typeof e ? B[G.call(e)] || "object": typeof e
		},
		globalEval: function(e) {
			var t, n = eval; (e = Q.trim(e)) && (1 === e.indexOf("use strict") ? (t = Z.createElement("script"), t.text = e, Z.head.appendChild(t).parentNode.removeChild(t)) : n(e))
		},
		camelCase: function(e) {
			return e.replace(ee, "ms-").replace(te, ne)
		},
		nodeName: function(e, t) {
			return e.nodeName && e.nodeName.toLowerCase() === t.toLowerCase()
		},
		each: function(e, t, i) {
			var r = 0,
			o = e.length,
			a = n(e);
			if (i) {
				if (a) for (; o > r && !1 !== t.apply(e[r], i); r++);
				else for (r in e) if (!1 === t.apply(e[r], i)) break
			} else if (a) for (; o > r && !1 !== t.call(e[r], r, e[r]); r++);
			else for (r in e) if (!1 === t.call(e[r], r, e[r])) break;
			return e
		},
		trim: function(e) {
			return null == e ? "": (e + "").replace(J, "")
		},
		makeArray: function(e, t) {
			var i = t || [];
			return null != e && (n(Object(e)) ? Q.merge(i, "string" == typeof e ? [e] : e) : R.call(i, e)),
			i
		},
		inArray: function(e, t, n) {
			return null == t ? -1 : U.call(t, e, n)
		},
		merge: function(e, t) {
			for (var n = +t.length,
			i = 0,
			r = e.length; n > i; i++) e[r++] = t[i];
			return e.length = r,
			e
		},
		grep: function(e, t, n) {
			for (var i = [], r = 0, o = e.length, a = !n; o > r; r++) ! t(e[r], r) !== a && i.push(e[r]);
			return i
		},
		map: function(e, t, i) {
			var r, o = 0,
			a = e.length,
			s = [];
			if (n(e)) for (; a > o; o++) null != (r = t(e[o], o, i)) && s.push(r);
			else for (o in e) null != (r = t(e[o], o, i)) && s.push(r);
			return Y.apply([], s)
		},
		guid: 1,
		proxy: function(e, t) {
			var n, i, r;
			return "string" == typeof t && (n = e[t], t = e, e = n),
			Q.isFunction(e) ? (i = F.call(arguments, 2), r = function() {
				return e.apply(t || this, i.concat(F.call(arguments)))
			},
			r.guid = e.guid = e.guid || Q.guid++, r) : void 0
		},
		now: Date.now,
		support: K
	}),
	Q.each("Boolean Number String Function Array Date RegExp Object Error".split(" "),
	function(e, t) {
		B["[object " + t + "]"] = t.toLowerCase()
	});
	var ie = function(e) {
		function t(e, t, n, i) {
			var r, o, a, s, c, d, h, f, p, m;
			if ((t ? t.ownerDocument || t: I) !== O && L(t), t = t || O, n = n || [], s = t.nodeType, "string" != typeof e || !e || 1 !== s && 9 !== s && 11 !== s) return n;
			if (!i && A) {
				if (11 !== s && (r = ge.exec(e))) if (a = r[1]) {
					if (9 === s) {
						if (! (o = t.getElementById(a)) || !o.parentNode) return n;
						if (o.id === a) return n.push(o),
						n
					} else if (t.ownerDocument && (o = t.ownerDocument.getElementById(a)) && j(t, o) && o.id === a) return n.push(o),
					n
				} else {
					if (r[2]) return Z.apply(n, t.getElementsByTagName(e)),
					n;
					if ((a = r[3]) && b.getElementsByClassName) return Z.apply(n, t.getElementsByClassName(a)),
					n
				}
				if (b.qsa && (!E || !E.test(e))) {
					if (f = h = P, p = t, m = 1 !== s && e, 1 === s && "object" !== t.nodeName.toLowerCase()) {
						for (d = _(e), (h = t.getAttribute("id")) ? f = h.replace(ye, "\\$&") : t.setAttribute("id", f), f = "[id='" + f + "'] ", c = d.length; c--;) d[c] = f + u(d[c]);
						p = ve.test(e) && l(t.parentNode) || t,
						m = d.join(",")
					}
					if (m) try {
						return Z.apply(n, p.querySelectorAll(m)),
						n
					} catch(e) {} finally {
						h || t.removeAttribute("id")
					}
				}
			}
			return S(e.replace(ae, "$1"), t, n, i)
		}
		function n() {
			function e(n, i) {
				return t.push(n + " ") > w.cacheLength && delete e[t.shift()],
				e[n + " "] = i
			}
			var t = [];
			return e
		}
		function i(e) {
			return e[P] = !0,
			e
		}
		function r(e) {
			var t = O.createElement("div");
			try {
				return !! e(t)
			} catch(e) {
				return ! 1
			} finally {
				t.parentNode && t.parentNode.removeChild(t),
				t = null
			}
		}
		function o(e, t) {
			for (var n = e.split("|"), i = e.length; i--;) w.attrHandle[n[i]] = t
		}
		function a(e, t) {
			var n = t && e,
			i = n && 1 === e.nodeType && 1 === t.nodeType && (~t.sourceIndex || U) - (~e.sourceIndex || U);
			if (i) return i;
			if (n) for (; n = n.nextSibling;) if (n === t) return - 1;
			return e ? 1 : -1
		}
		function s(e) {
			return i(function(t) {
				return t = +t,
				i(function(n, i) {
					for (var r, o = e([], n.length, t), a = o.length; a--;) n[r = o[a]] && (n[r] = !(i[r] = n[r]))
				})
			})
		}
		function l(e) {
			return e && void 0 !== e.getElementsByTagName && e
		}
		function c() {}
		function u(e) {
			for (var t = 0,
			n = e.length,
			i = ""; n > t; t++) i += e[t].value;
			return i
		}
		function d(e, t, n) {
			var i = t.dir,
			r = n && "parentNode" === i,
			o = W++;
			return t.first ?
			function(t, n, o) {
				for (; t = t[i];) if (1 === t.nodeType || r) return e(t, n, o)
			}: function(t, n, a) {
				var s, l, c = [z, o];
				if (a) {
					for (; t = t[i];) if ((1 === t.nodeType || r) && e(t, n, a)) return ! 0
				} else for (; t = t[i];) if (1 === t.nodeType || r) {
					if (l = t[P] || (t[P] = {}), (s = l[i]) && s[0] === z && s[1] === o) return c[2] = s[2];
					if (l[i] = c, c[2] = e(t, n, a)) return ! 0
				}
			}
		}
		function h(e) {
			return e.length > 1 ?
			function(t, n, i) {
				for (var r = e.length; r--;) if (!e[r](t, n, i)) return ! 1;
				return ! 0
			}: e[0]
		}
		function f(e, n, i) {
			for (var r = 0,
			o = n.length; o > r; r++) t(e, n[r], i);
			return i
		}
		function p(e, t, n, i, r) {
			for (var o, a = [], s = 0, l = e.length, c = null != t; l > s; s++)(o = e[s]) && (!n || n(o, i, r)) && (a.push(o), c && t.push(s));
			return a
		}
		function m(e, t, n, r, o, a) {
			return r && !r[P] && (r = m(r)),
			o && !o[P] && (o = m(o, a)),
			i(function(i, a, s, l) {
				var c, u, d, h = [],
				m = [],
				g = a.length,
				v = i || f(t || "*", s.nodeType ? [s] : s, []),
				y = !e || !i && t ? v: p(v, h, e, s, l),
				b = n ? o || (i ? e: g || r) ? [] : a: y;
				if (n && n(y, b, s, l), r) for (c = p(b, m), r(c, [], s, l), u = c.length; u--;)(d = c[u]) && (b[m[u]] = !(y[m[u]] = d));
				if (i) {
					if (o || e) {
						if (o) {
							for (c = [], u = b.length; u--;)(d = b[u]) && c.push(y[u] = d);
							o(null, b = [], c, l)
						}
						for (u = b.length; u--;)(d = b[u]) && (c = o ? Q(i, d) : h[u]) > -1 && (i[c] = !(a[c] = d))
					}
				} else b = p(b === a ? b.splice(g, b.length) : b),
				o ? o(null, a, b, l) : Z.apply(a, b)
			})
		}
		function g(e) {
			for (var t, n, i, r = e.length,
			o = w.relative[e[0].type], a = o || w.relative[" "], s = o ? 1 : 0, l = d(function(e) {
				return e === t
			},
			a, !0), c = d(function(e) {
				return Q(t, e) > -1
			},
			a, !0), f = [function(e, n, i) {
				var r = !o && (i || n !== M) || ((t = n).nodeType ? l(e, n, i) : c(e, n, i));
				return t = null,
				r
			}]; r > s; s++) if (n = w.relative[e[s].type]) f = [d(h(f), n)];
			else {
				if ((n = w.filter[e[s].type].apply(null, e[s].matches))[P]) {
					for (i = ++s; r > i && !w.relative[e[i].type]; i++);
					return m(s > 1 && h(f), s > 1 && u(e.slice(0, s - 1).concat({
						value: " " === e[s - 2].type ? "*": ""
					})).replace(ae, "$1"), n, i > s && g(e.slice(s, i)), r > i && g(e = e.slice(i)), r > i && u(e))
				}
				f.push(n)
			}
			return h(f)
		}
		function v(e, n) {
			var r = n.length > 0,
			o = e.length > 0,
			a = function(i, a, s, l, c) {
				var u, d, h, f = 0,
				m = "0",
				g = i && [],
				v = [],
				y = M,
				b = i || o && w.find.TAG("*", c),
				k = z += null == y ? 1 : Math.random() || .1,
				x = b.length;
				for (c && (M = a !== O && a); m !== x && null != (u = b[m]); m++) {
					if (o && u) {
						for (d = 0; h = e[d++];) if (h(u, a, s)) {
							l.push(u);
							break
						}
						c && (z = k)
					}
					r && ((u = !h && u) && f--, i && g.push(u))
				}
				if (f += m, r && m !== f) {
					for (d = 0; h = n[d++];) h(g, v, a, s);
					if (i) {
						if (f > 0) for (; m--;) g[m] || v[m] || (v[m] = V.call(l));
						v = p(v)
					}
					Z.apply(l, v),
					c && !i && v.length > 0 && f + n.length > 1 && t.uniqueSort(l)
				}
				return c && (z = k, M = y),
				g
			};
			return r ? i(a) : a
		}
		var y, b, w, k, x, _, C, S, M, T, D, L, O, N, A, E, $, q, j, P = "sizzle" + 1 * new Date,
		I = e.document,
		z = 0,
		W = 0,
		H = n(),
		F = n(),
		Y = n(),
		R = function(e, t) {
			return e === t && (D = !0),
			0
		},
		U = 1 << 31,
		B = {}.hasOwnProperty,
		G = [],
		V = G.pop,
		K = G.push,
		Z = G.push,
		X = G.slice,
		Q = function(e, t) {
			for (var n = 0,
			i = e.length; i > n; n++) if (e[n] === t) return n;
			return - 1
		},
		J = "checked|selected|async|autofocus|autoplay|controls|defer|disabled|hidden|ismap|loop|multiple|open|readonly|required|scoped",
		ee = "[\\x20\\t\\r\\n\\f]",
		te = "(?:\\\\.|[\\w-]|[^\\x00-\\xa0])+",
		ne = te.replace("w", "w#"),
		ie = "\\[" + ee + "*(" + te + ")(?:" + ee + "*([*^$|!~]?=)" + ee + "*(?:'((?:\\\\.|[^\\\\'])*)'|\"((?:\\\\.|[^\\\\\"])*)\"|(" + ne + "))|)" + ee + "*\\]",
		re = ":(" + te + ")(?:\\((('((?:\\\\.|[^\\\\'])*)'|\"((?:\\\\.|[^\\\\\"])*)\")|((?:\\\\.|[^\\\\()[\\]]|" + ie + ")*)|.*)\\)|)",
		oe = new RegExp(ee + "+", "g"),
		ae = new RegExp("^" + ee + "+|((?:^|[^\\\\])(?:\\\\.)*)" + ee + "+$", "g"),
		se = new RegExp("^" + ee + "*," + ee + "*"),
		le = new RegExp("^" + ee + "*([>+~]|" + ee + ")" + ee + "*"),
		ce = new RegExp("=" + ee + "*([^\\]'\"]*?)" + ee + "*\\]", "g"),
		ue = new RegExp(re),
		de = new RegExp("^" + ne + "$"),
		he = {
			ID: new RegExp("^#(" + te + ")"),
			CLASS: new RegExp("^\\.(" + te + ")"),
			TAG: new RegExp("^(" + te.replace("w", "w*") + ")"),
			ATTR: new RegExp("^" + ie),
			PSEUDO: new RegExp("^" + re),
			CHILD: new RegExp("^:(only|first|last|nth|nth-last)-(child|of-type)(?:\\(" + ee + "*(even|odd|(([+-]|)(\\d*)n|)" + ee + "*(?:([+-]|)" + ee + "*(\\d+)|))" + ee + "*\\)|)", "i"),
			bool: new RegExp("^(?:" + J + ")$", "i"),
			needsContext: new RegExp("^" + ee + "*[>+~]|:(even|odd|eq|gt|lt|nth|first|last)(?:\\(" + ee + "*((?:-\\d)?\\d*)" + ee + "*\\)|)(?=[^-]|$)", "i")
		},
		fe = /^(?:input|select|textarea|button)$/i,
		pe = /^h\d$/i,
		me = /^[^{]+\{\s*\[native \w/,
		ge = /^(?:#([\w-]+)|(\w+)|\.([\w-]+))$/,
		ve = /[+~]/,
		ye = /'|\\/g,
		be = new RegExp("\\\\([\\da-f]{1,6}" + ee + "?|(" + ee + ")|.)", "ig"),
		we = function(e, t, n) {
			var i = "0x" + t - 65536;
			return i !== i || n ? t: 0 > i ? String.fromCharCode(i + 65536) : String.fromCharCode(i >> 10 | 55296, 1023 & i | 56320)
		},
		ke = function() {
			L()
		};
		try {
			Z.apply(G = X.call(I.childNodes), I.childNodes),
			G[I.childNodes.length].nodeType
		} catch(e) {
			Z = {
				apply: G.length ?
				function(e, t) {
					K.apply(e, X.call(t))
				}: function(e, t) {
					for (var n = e.length,
					i = 0; e[n++] = t[i++];);
					e.length = n - 1
				}
			}
		}
		b = t.support = {},
		x = t.isXML = function(e) {
			var t = e && (e.ownerDocument || e).documentElement;
			return !! t && "HTML" !== t.nodeName
		},
		L = t.setDocument = function(e) {
			var t, n, i = e ? e.ownerDocument || e: I;
			return i !== O && 9 === i.nodeType && i.documentElement ? (O = i, N = i.documentElement, (n = i.defaultView) && n !== n.top && (n.addEventListener ? n.addEventListener("unload", ke, !1) : n.attachEvent && n.attachEvent("onunload", ke)), A = !x(i), b.attributes = r(function(e) {
				return e.className = "i",
				!e.getAttribute("className")
			}), b.getElementsByTagName = r(function(e) {
				return e.appendChild(i.createComment("")),
				!e.getElementsByTagName("*").length
			}), b.getElementsByClassName = me.test(i.getElementsByClassName), b.getById = r(function(e) {
				return N.appendChild(e).id = P,
				!i.getElementsByName || !i.getElementsByName(P).length
			}), b.getById ? (w.find.ID = function(e, t) {
				if (void 0 !== t.getElementById && A) {
					var n = t.getElementById(e);
					return n && n.parentNode ? [n] : []
				}
			},
			w.filter.ID = function(e) {
				var t = e.replace(be, we);
				return function(e) {
					return e.getAttribute("id") === t
				}
			}) : (delete w.find.ID, w.filter.ID = function(e) {
				var t = e.replace(be, we);
				return function(e) {
					var n = void 0 !== e.getAttributeNode && e.getAttributeNode("id");
					return n && n.value === t
				}
			}), w.find.TAG = b.getElementsByTagName ?
			function(e, t) {
				return void 0 !== t.getElementsByTagName ? t.getElementsByTagName(e) : b.qsa ? t.querySelectorAll(e) : void 0
			}: function(e, t) {
				var n, i = [],
				r = 0,
				o = t.getElementsByTagName(e);
				if ("*" === e) {
					for (; n = o[r++];) 1 === n.nodeType && i.push(n);
					return i
				}
				return o
			},
			w.find.CLASS = b.getElementsByClassName &&
			function(e, t) {
				return A ? t.getElementsByClassName(e) : void 0
			},
			$ = [], E = [], (b.qsa = me.test(i.querySelectorAll)) && (r(function(e) {
				N.appendChild(e).innerHTML = "<a id='" + P + "'></a><select id='" + P + "-\f]' msallowcapture=''><option selected=''></option></select>",
				e.querySelectorAll("[msallowcapture^='']").length && E.push("[*^$]=" + ee + "*(?:''|\"\")"),
				e.querySelectorAll("[selected]").length || E.push("\\[" + ee + "*(?:value|" + J + ")"),
				e.querySelectorAll("[id~=" + P + "-]").length || E.push("~="),
				e.querySelectorAll(":checked").length || E.push(":checked"),
				e.querySelectorAll("a#" + P + "+*").length || E.push(".#.+[+~]")
			}), r(function(e) {
				var t = i.createElement("input");
				t.setAttribute("type", "hidden"),
				e.appendChild(t).setAttribute("name", "D"),
				e.querySelectorAll("[name=d]").length && E.push("name" + ee + "*[*^$|!~]?="),
				e.querySelectorAll(":enabled").length || E.push(":enabled", ":disabled"),
				e.querySelectorAll("*,:x"),
				E.push(",.*:")
			})), (b.matchesSelector = me.test(q = N.matches || N.webkitMatchesSelector || N.mozMatchesSelector || N.oMatchesSelector || N.msMatchesSelector)) && r(function(e) {
				b.disconnectedMatch = q.call(e, "div"),
				q.call(e, "[s!='']:x"),
				$.push("!=", re)
			}), E = E.length && new RegExp(E.join("|")), $ = $.length && new RegExp($.join("|")), t = me.test(N.compareDocumentPosition), j = t || me.test(N.contains) ?
			function(e, t) {
				var n = 9 === e.nodeType ? e.documentElement: e,
				i = t && t.parentNode;
				return e === i || !(!i || 1 !== i.nodeType || !(n.contains ? n.contains(i) : e.compareDocumentPosition && 16 & e.compareDocumentPosition(i)))
			}: function(e, t) {
				if (t) for (; t = t.parentNode;) if (t === e) return ! 0;
				return ! 1
			},
			R = t ?
			function(e, t) {
				if (e === t) return D = !0,
				0;
				var n = !e.compareDocumentPosition - !t.compareDocumentPosition;
				return n || (1 & (n = (e.ownerDocument || e) === (t.ownerDocument || t) ? e.compareDocumentPosition(t) : 1) || !b.sortDetached && t.compareDocumentPosition(e) === n ? e === i || e.ownerDocument === I && j(I, e) ? -1 : t === i || t.ownerDocument === I && j(I, t) ? 1 : T ? Q(T, e) - Q(T, t) : 0 : 4 & n ? -1 : 1)
			}: function(e, t) {
				if (e === t) return D = !0,
				0;
				var n, r = 0,
				o = e.parentNode,
				s = t.parentNode,
				l = [e],
				c = [t];
				if (!o || !s) return e === i ? -1 : t === i ? 1 : o ? -1 : s ? 1 : T ? Q(T, e) - Q(T, t) : 0;
				if (o === s) return a(e, t);
				for (n = e; n = n.parentNode;) l.unshift(n);
				for (n = t; n = n.parentNode;) c.unshift(n);
				for (; l[r] === c[r];) r++;
				return r ? a(l[r], c[r]) : l[r] === I ? -1 : c[r] === I ? 1 : 0
			},
			i) : O
		},
		t.matches = function(e, n) {
			return t(e, null, null, n)
		},
		t.matchesSelector = function(e, n) {
			if ((e.ownerDocument || e) !== O && L(e), n = n.replace(ce, "='$1']"), !(!b.matchesSelector || !A || $ && $.test(n) || E && E.test(n))) try {
				var i = q.call(e, n);
				if (i || b.disconnectedMatch || e.document && 11 !== e.document.nodeType) return i
			} catch(e) {}
			return t(n, O, null, [e]).length > 0
		},
		t.contains = function(e, t) {
			return (e.ownerDocument || e) !== O && L(e),
			j(e, t)
		},
		t.attr = function(e, t) { (e.ownerDocument || e) !== O && L(e);
			var n = w.attrHandle[t.toLowerCase()],
			i = n && B.call(w.attrHandle, t.toLowerCase()) ? n(e, t, !A) : void 0;
			return void 0 !== i ? i: b.attributes || !A ? e.getAttribute(t) : (i = e.getAttributeNode(t)) && i.specified ? i.value: null
		},
		t.error = function(e) {
			throw new Error("Syntax error, unrecognized expression: " + e)
		},
		t.uniqueSort = function(e) {
			var t, n = [],
			i = 0,
			r = 0;
			if (D = !b.detectDuplicates, T = !b.sortStable && e.slice(0), e.sort(R), D) {
				for (; t = e[r++];) t === e[r] && (i = n.push(r));
				for (; i--;) e.splice(n[i], 1)
			}
			return T = null,
			e
		},
		k = t.getText = function(e) {
			var t, n = "",
			i = 0,
			r = e.nodeType;
			if (r) {
				if (1 === r || 9 === r || 11 === r) {
					if ("string" == typeof e.textContent) return e.textContent;
					for (e = e.firstChild; e; e = e.nextSibling) n += k(e)
				} else if (3 === r || 4 === r) return e.nodeValue
			} else for (; t = e[i++];) n += k(t);
			return n
		},
		(w = t.selectors = {
			cacheLength: 50,
			createPseudo: i,
			match: he,
			attrHandle: {},
			find: {},
			relative: {
				">": {
					dir: "parentNode",
					first: !0
				},
				" ": {
					dir: "parentNode"
				},
				"+": {
					dir: "previousSibling",
					first: !0
				},
				"~": {
					dir: "previousSibling"
				}
			},
			preFilter: {
				ATTR: function(e) {
					return e[1] = e[1].replace(be, we),
					e[3] = (e[3] || e[4] || e[5] || "").replace(be, we),
					"~=" === e[2] && (e[3] = " " + e[3] + " "),
					e.slice(0, 4)
				},
				CHILD: function(e) {
					return e[1] = e[1].toLowerCase(),
					"nth" === e[1].slice(0, 3) ? (e[3] || t.error(e[0]), e[4] = +(e[4] ? e[5] + (e[6] || 1) : 2 * ("even" === e[3] || "odd" === e[3])), e[5] = +(e[7] + e[8] || "odd" === e[3])) : e[3] && t.error(e[0]),
					e
				},
				PSEUDO: function(e) {
					var t, n = !e[6] && e[2];
					return he.CHILD.test(e[0]) ? null: (e[3] ? e[2] = e[4] || e[5] || "": n && ue.test(n) && (t = _(n, !0)) && (t = n.indexOf(")", n.length - t) - n.length) && (e[0] = e[0].slice(0, t), e[2] = n.slice(0, t)), e.slice(0, 3))
				}
			},
			filter: {
				TAG: function(e) {
					var t = e.replace(be, we).toLowerCase();
					return "*" === e ?
					function() {
						return ! 0
					}: function(e) {
						return e.nodeName && e.nodeName.toLowerCase() === t
					}
				},
				CLASS: function(e) {
					var t = H[e + " "];
					return t || (t = new RegExp("(^|" + ee + ")" + e + "(" + ee + "|$)")) && H(e,
					function(e) {
						return t.test("string" == typeof e.className && e.className || void 0 !== e.getAttribute && e.getAttribute("class") || "")
					})
				},
				ATTR: function(e, n, i) {
					return function(r) {
						var o = t.attr(r, e);
						return null == o ? "!=" === n: !n || (o += "", "=" === n ? o === i: "!=" === n ? o !== i: "^=" === n ? i && 0 === o.indexOf(i) : "*=" === n ? i && o.indexOf(i) > -1 : "$=" === n ? i && o.slice( - i.length) === i: "~=" === n ? (" " + o.replace(oe, " ") + " ").indexOf(i) > -1 : "|=" === n && (o === i || o.slice(0, i.length + 1) === i + "-"))
					}
				},
				CHILD: function(e, t, n, i, r) {
					var o = "nth" !== e.slice(0, 3),
					a = "last" !== e.slice( - 4),
					s = "of-type" === t;
					return 1 === i && 0 === r ?
					function(e) {
						return !! e.parentNode
					}: function(t, n, l) {
						var c, u, d, h, f, p, m = o !== a ? "nextSibling": "previousSibling",
						g = t.parentNode,
						v = s && t.nodeName.toLowerCase(),
						y = !l && !s;
						if (g) {
							if (o) {
								for (; m;) {
									for (d = t; d = d[m];) if (s ? d.nodeName.toLowerCase() === v: 1 === d.nodeType) return ! 1;
									p = m = "only" === e && !p && "nextSibling"
								}
								return ! 0
							}
							if (p = [a ? g.firstChild: g.lastChild], a && y) {
								for (f = (c = (u = g[P] || (g[P] = {}))[e] || [])[0] === z && c[1], h = c[0] === z && c[2], d = f && g.childNodes[f]; d = ++f && d && d[m] || (h = f = 0) || p.pop();) if (1 === d.nodeType && ++h && d === t) {
									u[e] = [z, f, h];
									break
								}
							} else if (y && (c = (t[P] || (t[P] = {}))[e]) && c[0] === z) h = c[1];
							else for (; (d = ++f && d && d[m] || (h = f = 0) || p.pop()) && ((s ? d.nodeName.toLowerCase() !== v: 1 !== d.nodeType) || !++h || (y && ((d[P] || (d[P] = {}))[e] = [z, h]), d !== t)););
							return (h -= r) === i || h % i == 0 && h / i >= 0
						}
					}
				},
				PSEUDO: function(e, n) {
					var r, o = w.pseudos[e] || w.setFilters[e.toLowerCase()] || t.error("unsupported pseudo: " + e);
					return o[P] ? o(n) : o.length > 1 ? (r = [e, e, "", n], w.setFilters.hasOwnProperty(e.toLowerCase()) ? i(function(e, t) {
						for (var i, r = o(e, n), a = r.length; a--;) i = Q(e, r[a]),
						e[i] = !(t[i] = r[a])
					}) : function(e) {
						return o(e, 0, r)
					}) : o
				}
			},
			pseudos: {
				not: i(function(e) {
					var t = [],
					n = [],
					r = C(e.replace(ae, "$1"));
					return r[P] ? i(function(e, t, n, i) {
						for (var o, a = r(e, null, i, []), s = e.length; s--;)(o = a[s]) && (e[s] = !(t[s] = o))
					}) : function(e, i, o) {
						return t[0] = e,
						r(t, null, o, n),
						t[0] = null,
						!n.pop()
					}
				}),
				has: i(function(e) {
					return function(n) {
						return t(e, n).length > 0
					}
				}),
				contains: i(function(e) {
					return e = e.replace(be, we),
					function(t) {
						return (t.textContent || t.innerText || k(t)).indexOf(e) > -1
					}
				}),
				lang: i(function(e) {
					return de.test(e || "") || t.error("unsupported lang: " + e),
					e = e.replace(be, we).toLowerCase(),
					function(t) {
						var n;
						do {
							if (n = A ? t.lang: t.getAttribute("xml:lang") || t.getAttribute("lang")) return (n = n.toLowerCase()) === e || 0 === n.indexOf(e + "-")
						} while (( t = t . parentNode ) && 1 === t.nodeType);
						return ! 1
					}
				}),
				target: function(t) {
					var n = e.location && e.location.hash;
					return n && n.slice(1) === t.id
				},
				root: function(e) {
					return e === N
				},
				focus: function(e) {
					return e === O.activeElement && (!O.hasFocus || O.hasFocus()) && !!(e.type || e.href || ~e.tabIndex)
				},
				enabled: function(e) {
					return ! 1 === e.disabled
				},
				disabled: function(e) {
					return ! 0 === e.disabled
				},
				checked: function(e) {
					var t = e.nodeName.toLowerCase();
					return "input" === t && !!e.checked || "option" === t && !!e.selected
				},
				selected: function(e) {
					return e.parentNode && e.parentNode.selectedIndex,
					!0 === e.selected
				},
				empty: function(e) {
					for (e = e.firstChild; e; e = e.nextSibling) if (e.nodeType < 6) return ! 1;
					return ! 0
				},
				parent: function(e) {
					return ! w.pseudos.empty(e)
				},
				header: function(e) {
					return pe.test(e.nodeName)
				},
				input: function(e) {
					return fe.test(e.nodeName)
				},
				button: function(e) {
					var t = e.nodeName.toLowerCase();
					return "input" === t && "button" === e.type || "button" === t
				},
				text: function(e) {
					var t;
					return "input" === e.nodeName.toLowerCase() && "text" === e.type && (null == (t = e.getAttribute("type")) || "text" === t.toLowerCase())
				},
				first: s(function() {
					return [0]
				}),
				last: s(function(e, t) {
					return [t - 1]
				}),
				eq: s(function(e, t, n) {
					return [0 > n ? n + t: n]
				}),
				even: s(function(e, t) {
					for (var n = 0; t > n; n += 2) e.push(n);
					return e
				}),
				odd: s(function(e, t) {
					for (var n = 1; t > n; n += 2) e.push(n);
					return e
				}),
				lt: s(function(e, t, n) {
					for (var i = 0 > n ? n + t: n; --i >= 0;) e.push(i);
					return e
				}),
				gt: s(function(e, t, n) {
					for (var i = 0 > n ? n + t: n; ++i < t;) e.push(i);
					return e
				})
			}
		}).pseudos.nth = w.pseudos.eq;
		for (y in {
			radio: !0,
			checkbox: !0,
			file: !0,
			password: !0,
			image: !0
		}) w.pseudos[y] = function(e) {
			return function(t) {
				return "input" === t.nodeName.toLowerCase() && t.type === e
			}
		} (y);
		for (y in {
			submit: !0,
			reset: !0
		}) w.pseudos[y] = function(e) {
			return function(t) {
				var n = t.nodeName.toLowerCase();
				return ("input" === n || "button" === n) && t.type === e
			}
		} (y);
		return c.prototype = w.filters = w.pseudos,
		w.setFilters = new c,
		_ = t.tokenize = function(e, n) {
			var i, r, o, a, s, l, c, u = F[e + " "];
			if (u) return n ? 0 : u.slice(0);
			for (s = e, l = [], c = w.preFilter; s;) { (!i || (r = se.exec(s))) && (r && (s = s.slice(r[0].length) || s), l.push(o = [])),
				i = !1,
				(r = le.exec(s)) && (i = r.shift(), o.push({
					value: i,
					type: r[0].replace(ae, " ")
				}), s = s.slice(i.length));
				for (a in w.filter) ! (r = he[a].exec(s)) || c[a] && !(r = c[a](r)) || (i = r.shift(), o.push({
					value: i,
					type: a,
					matches: r
				}), s = s.slice(i.length));
				if (!i) break
			}
			return n ? s.length: s ? t.error(e) : F(e, l).slice(0)
		},
		C = t.compile = function(e, t) {
			var n, i = [],
			r = [],
			o = Y[e + " "];
			if (!o) {
				for (t || (t = _(e)), n = t.length; n--;)(o = g(t[n]))[P] ? i.push(o) : r.push(o); (o = Y(e, v(r, i))).selector = e
			}
			return o
		},
		S = t.select = function(e, t, n, i) {
			var r, o, a, s, c, d = "function" == typeof e && e,
			h = !i && _(e = d.selector || e);
			if (n = n || [], 1 === h.length) {
				if ((o = h[0] = h[0].slice(0)).length > 2 && "ID" === (a = o[0]).type && b.getById && 9 === t.nodeType && A && w.relative[o[1].type]) {
					if (! (t = (w.find.ID(a.matches[0].replace(be, we), t) || [])[0])) return n;
					d && (t = t.parentNode),
					e = e.slice(o.shift().value.length)
				}
				for (r = he.needsContext.test(e) ? 0 : o.length; r--&&(a = o[r], !w.relative[s = a.type]);) if ((c = w.find[s]) && (i = c(a.matches[0].replace(be, we), ve.test(o[0].type) && l(t.parentNode) || t))) {
					if (o.splice(r, 1), !(e = i.length && u(o))) return Z.apply(n, i),
					n;
					break
				}
			}
			return (d || C(e, h))(i, t, !A, n, ve.test(e) && l(t.parentNode) || t),
			n
		},
		b.sortStable = P.split("").sort(R).join("") === P,
		b.detectDuplicates = !!D,
		L(),
		b.sortDetached = r(function(e) {
			return 1 & e.compareDocumentPosition(O.createElement("div"))
		}),
		r(function(e) {
			return e.innerHTML = "<a href='#'></a>",
			"#" === e.firstChild.getAttribute("href")
		}) || o("type|href|height|width",
		function(e, t, n) {
			return n ? void 0 : e.getAttribute(t, "type" === t.toLowerCase() ? 1 : 2)
		}),
		b.attributes && r(function(e) {
			return e.innerHTML = "<input/>",
			e.firstChild.setAttribute("value", ""),
			"" === e.firstChild.getAttribute("value")
		}) || o("value",
		function(e, t, n) {
			return n || "input" !== e.nodeName.toLowerCase() ? void 0 : e.defaultValue
		}),
		r(function(e) {
			return null == e.getAttribute("disabled")
		}) || o(J,
		function(e, t, n) {
			var i;
			return n ? void 0 : !0 === e[t] ? t.toLowerCase() : (i = e.getAttributeNode(t)) && i.specified ? i.value: null
		}),
		t
	} (e);
	Q.find = ie,
	Q.expr = ie.selectors,
	Q.expr[":"] = Q.expr.pseudos,
	Q.unique = ie.uniqueSort,
	Q.text = ie.getText,
	Q.isXMLDoc = ie.isXML,
	Q.contains = ie.contains;
	var re = Q.expr.match.needsContext,
	oe = /^<(\w+)\s*\/?>(?:<\/\1>|)$/,
	ae = /^.[^:#\[\.,]*$/;
	Q.filter = function(e, t, n) {
		var i = t[0];
		return n && (e = ":not(" + e + ")"),
		1 === t.length && 1 === i.nodeType ? Q.find.matchesSelector(i, e) ? [i] : [] : Q.find.matches(e, Q.grep(t,
		function(e) {
			return 1 === e.nodeType
		}))
	},
	Q.fn.extend({
		find: function(e) {
			var t, n = this.length,
			i = [],
			r = this;
			if ("string" != typeof e) return this.pushStack(Q(e).filter(function() {
				for (t = 0; n > t; t++) if (Q.contains(r[t], this)) return ! 0
			}));
			for (t = 0; n > t; t++) Q.find(e, r[t], i);
			return i = this.pushStack(n > 1 ? Q.unique(i) : i),
			i.selector = this.selector ? this.selector + " " + e: e,
			i
		},
		filter: function(e) {
			return this.pushStack(i(this, e || [], !1))
		},
		not: function(e) {
			return this.pushStack(i(this, e || [], !0))
		},
		is: function(e) {
			return !! i(this, "string" == typeof e && re.test(e) ? Q(e) : e || [], !1).length
		}
	});
	var se, le = /^(?:\s*(<[\w\W]+>)[^>]*|#([\w-]*))$/; (Q.fn.init = function(e, t) {
		var n, i;
		if (!e) return this;
		if ("string" == typeof e) {
			if (! (n = "<" === e[0] && ">" === e[e.length - 1] && e.length >= 3 ? [null, e, null] : le.exec(e)) || !n[1] && t) return ! t || t.jquery ? (t || se).find(e) : this.constructor(t).find(e);
			if (n[1]) {
				if (t = t instanceof Q ? t[0] : t, Q.merge(this, Q.parseHTML(n[1], t && t.nodeType ? t.ownerDocument || t: Z, !0)), oe.test(n[1]) && Q.isPlainObject(t)) for (n in t) Q.isFunction(this[n]) ? this[n](t[n]) : this.attr(n, t[n]);
				return this
			}
			return (i = Z.getElementById(n[2])) && i.parentNode && (this.length = 1, this[0] = i),
			this.context = Z,
			this.selector = e,
			this
		}
		return e.nodeType ? (this.context = this[0] = e, this.length = 1, this) : Q.isFunction(e) ? void 0 !== se.ready ? se.ready(e) : e(Q) : (void 0 !== e.selector && (this.selector = e.selector, this.context = e.context), Q.makeArray(e, this))
	}).prototype = Q.fn,
	se = Q(Z);
	var ce = /^(?:parents|prev(?:Until|All))/,
	ue = {
		children: !0,
		contents: !0,
		next: !0,
		prev: !0
	};
	Q.extend({
		dir: function(e, t, n) {
			for (var i = [], r = void 0 !== n; (e = e[t]) && 9 !== e.nodeType;) if (1 === e.nodeType) {
				if (r && Q(e).is(n)) break;
				i.push(e)
			}
			return i
		},
		sibling: function(e, t) {
			for (var n = []; e; e = e.nextSibling) 1 === e.nodeType && e !== t && n.push(e);
			return n
		}
	}),
	Q.fn.extend({
		has: function(e) {
			var t = Q(e, this),
			n = t.length;
			return this.filter(function() {
				for (var e = 0; n > e; e++) if (Q.contains(this, t[e])) return ! 0
			})
		},
		closest: function(e, t) {
			for (var n, i = 0,
			r = this.length,
			o = [], a = re.test(e) || "string" != typeof e ? Q(e, t || this.context) : 0; r > i; i++) for (n = this[i]; n && n !== t; n = n.parentNode) if (n.nodeType < 11 && (a ? a.index(n) > -1 : 1 === n.nodeType && Q.find.matchesSelector(n, e))) {
				o.push(n);
				break
			}
			return this.pushStack(o.length > 1 ? Q.unique(o) : o)
		},
		index: function(e) {
			return e ? "string" == typeof e ? U.call(Q(e), this[0]) : U.call(this, e.jquery ? e[0] : e) : this[0] && this[0].parentNode ? this.first().prevAll().length: -1
		},
		add: function(e, t) {
			return this.pushStack(Q.unique(Q.merge(this.get(), Q(e, t))))
		},
		addBack: function(e) {
			return this.add(null == e ? this.prevObject: this.prevObject.filter(e))
		}
	}),
	Q.each({
		parent: function(e) {
			var t = e.parentNode;
			return t && 11 !== t.nodeType ? t: null
		},
		parents: function(e) {
			return Q.dir(e, "parentNode")
		},
		parentsUntil: function(e, t, n) {
			return Q.dir(e, "parentNode", n)
		},
		next: function(e) {
			return r(e, "nextSibling")
		},
		prev: function(e) {
			return r(e, "previousSibling")
		},
		nextAll: function(e) {
			return Q.dir(e, "nextSibling")
		},
		prevAll: function(e) {
			return Q.dir(e, "previousSibling")
		},
		nextUntil: function(e, t, n) {
			return Q.dir(e, "nextSibling", n)
		},
		prevUntil: function(e, t, n) {
			return Q.dir(e, "previousSibling", n)
		},
		siblings: function(e) {
			return Q.sibling((e.parentNode || {}).firstChild, e)
		},
		children: function(e) {
			return Q.sibling(e.firstChild)
		},
		contents: function(e) {
			return e.contentDocument || Q.merge([], e.childNodes)
		}
	},
	function(e, t) {
		Q.fn[e] = function(n, i) {
			var r = Q.map(this, t, n);
			return "Until" !== e.slice( - 5) && (i = n),
			i && "string" == typeof i && (r = Q.filter(i, r)),
			this.length > 1 && (ue[e] || Q.unique(r), ce.test(e) && r.reverse()),
			this.pushStack(r)
		}
	});
	var de = /\S+/g,
	he = {};
	Q.Callbacks = function(e) {
		var t, n, i, r, a, s, l = [],
		c = !(e = "string" == typeof e ? he[e] || o(e) : Q.extend({},
		e)).once && [],
		u = function(o) {
			for (t = e.memory && o, n = !0, s = r || 0, r = 0, a = l.length, i = !0; l && a > s; s++) if (!1 === l[s].apply(o[0], o[1]) && e.stopOnFalse) {
				t = !1;
				break
			}
			i = !1,
			l && (c ? c.length && u(c.shift()) : t ? l = [] : d.disable())
		},
		d = {
			add: function() {
				if (l) {
					var n = l.length; !
					function t(n) {
						Q.each(n,
						function(n, i) {
							var r = Q.type(i);
							"function" === r ? e.unique && d.has(i) || l.push(i) : i && i.length && "string" !== r && t(i)
						})
					} (arguments),
					i ? a = l.length: t && (r = n, u(t))
				}
				return this
			},
			remove: function() {
				return l && Q.each(arguments,
				function(e, t) {
					for (var n; (n = Q.inArray(t, l, n)) > -1;) l.splice(n, 1),
					i && (a >= n && a--, s >= n && s--)
				}),
				this
			},
			has: function(e) {
				return e ? Q.inArray(e, l) > -1 : !(!l || !l.length)
			},
			empty: function() {
				return l = [],
				a = 0,
				this
			},
			disable: function() {
				return l = c = t = void 0,
				this
			},
			disabled: function() {
				return ! l
			},
			lock: function() {
				return c = void 0,
				t || d.disable(),
				this
			},
			locked: function() {
				return ! c
			},
			fireWith: function(e, t) {
				return ! l || n && !c || (t = t || [], t = [e, t.slice ? t.slice() : t], i ? c.push(t) : u(t)),
				this
			},
			fire: function() {
				return d.fireWith(this, arguments),
				this
			},
			fired: function() {
				return !! n
			}
		};
		return d
	},
	Q.extend({
		Deferred: function(e) {
			var t = [["resolve", "done", Q.Callbacks("once memory"), "resolved"], ["reject", "fail", Q.Callbacks("once memory"), "rejected"], ["notify", "progress", Q.Callbacks("memory")]],
			n = "pending",
			i = {
				state: function() {
					return n
				},
				always: function() {
					return r.done(arguments).fail(arguments),
					this
				},
				then: function() {
					var e = arguments;
					return Q.Deferred(function(n) {
						Q.each(t,
						function(t, o) {
							var a = Q.isFunction(e[t]) && e[t];
							r[o[1]](function() {
								var e = a && a.apply(this, arguments);
								e && Q.isFunction(e.promise) ? e.promise().done(n.resolve).fail(n.reject).progress(n.notify) : n[o[0] + "With"](this === i ? n.promise() : this, a ? [e] : arguments)
							})
						}),
						e = null
					}).promise()
				},
				promise: function(e) {
					return null != e ? Q.extend(e, i) : i
				}
			},
			r = {};
			return i.pipe = i.then,
			Q.each(t,
			function(e, o) {
				var a = o[2],
				s = o[3];
				i[o[1]] = a.add,
				s && a.add(function() {
					n = s
				},
				t[1 ^ e][2].disable, t[2][2].lock),
				r[o[0]] = function() {
					return r[o[0] + "With"](this === r ? i: this, arguments),
					this
				},
				r[o[0] + "With"] = a.fireWith
			}),
			i.promise(r),
			e && e.call(r, r),
			r
		},
		when: function(e) {
			var t, n, i, r = 0,
			o = F.call(arguments),
			a = o.length,
			s = 1 !== a || e && Q.isFunction(e.promise) ? a: 0,
			l = 1 === s ? e: Q.Deferred(),
			c = function(e, n, i) {
				return function(r) {
					n[e] = this,
					i[e] = arguments.length > 1 ? F.call(arguments) : r,
					i === t ? l.notifyWith(n, i) : --s || l.resolveWith(n, i)
				}
			};
			if (a > 1) for (t = new Array(a), n = new Array(a), i = new Array(a); a > r; r++) o[r] && Q.isFunction(o[r].promise) ? o[r].promise().done(c(r, i, o)).fail(l.reject).progress(c(r, n, t)) : --s;
			return s || l.resolveWith(i, o),
			l.promise()
		}
	});
	var fe;
	Q.fn.ready = function(e) {
		return Q.ready.promise().done(e),
		this
	},
	Q.extend({
		isReady: !1,
		readyWait: 1,
		holdReady: function(e) {
			e ? Q.readyWait++:Q.ready(!0)
		},
		ready: function(e) { (!0 === e ? --Q.readyWait: Q.isReady) || (Q.isReady = !0, !0 !== e && --Q.readyWait > 0 || (fe.resolveWith(Z, [Q]), Q.fn.triggerHandler && (Q(Z).triggerHandler("ready"), Q(Z).off("ready"))))
		}
	}),
	Q.ready.promise = function(t) {
		return fe || (fe = Q.Deferred(), "complete" === Z.readyState ? setTimeout(Q.ready) : (Z.addEventListener("DOMContentLoaded", a, !1), e.addEventListener("load", a, !1))),
		fe.promise(t)
	},
	Q.ready.promise();
	var pe = Q.access = function(e, t, n, i, r, o, a) {
		var s = 0,
		l = e.length,
		c = null == n;
		if ("object" === Q.type(n)) {
			r = !0;
			for (s in n) Q.access(e, t, s, n[s], !0, o, a)
		} else if (void 0 !== i && (r = !0, Q.isFunction(i) || (a = !0), c && (a ? (t.call(e, i), t = null) : (c = t, t = function(e, t, n) {
			return c.call(Q(e), n)
		})), t)) for (; l > s; s++) t(e[s], n, a ? i: i.call(e[s], s, t(e[s], n)));
		return r ? e: c ? t.call(e) : l ? t(e[0], n) : o
	};
	Q.acceptData = function(e) {
		return 1 === e.nodeType || 9 === e.nodeType || !+e.nodeType
	},
	s.uid = 1,
	s.accepts = Q.acceptData,
	s.prototype = {
		key: function(e) {
			if (!s.accepts(e)) return 0;
			var t = {},
			n = e[this.expando];
			if (!n) {
				n = s.uid++;
				try {
					t[this.expando] = {
						value: n
					},
					Object.defineProperties(e, t)
				} catch(i) {
					t[this.expando] = n,
					Q.extend(e, t)
				}
			}
			return this.cache[n] || (this.cache[n] = {}),
			n
		},
		set: function(e, t, n) {
			var i, r = this.key(e),
			o = this.cache[r];
			if ("string" == typeof t) o[t] = n;
			else if (Q.isEmptyObject(o)) Q.extend(this.cache[r], t);
			else for (i in t) o[i] = t[i];
			return o
		},
		get: function(e, t) {
			var n = this.cache[this.key(e)];
			return void 0 === t ? n: n[t]
		},
		access: function(e, t, n) {
			var i;
			return void 0 === t || t && "string" == typeof t && void 0 === n ? void 0 !== (i = this.get(e, t)) ? i: this.get(e, Q.camelCase(t)) : (this.set(e, t, n), void 0 !== n ? n: t)
		},
		remove: function(e, t) {
			var n, i, r, o = this.key(e),
			a = this.cache[o];
			if (void 0 === t) this.cache[o] = {};
			else {
				Q.isArray(t) ? i = t.concat(t.map(Q.camelCase)) : (r = Q.camelCase(t), t in a ? i = [t, r] : (i = r, i = i in a ? [i] : i.match(de) || [])),
				n = i.length;
				for (; n--;) delete a[i[n]]
			}
		},
		hasData: function(e) {
			return ! Q.isEmptyObject(this.cache[e[this.expando]] || {})
		},
		discard: function(e) {
			e[this.expando] && delete this.cache[e[this.expando]]
		}
	};
	var me = new s,
	ge = new s,
	ve = /^(?:\{[\w\W]*\}|\[[\w\W]*\])$/,
	ye = /([A-Z])/g;
	Q.extend({
		hasData: function(e) {
			return ge.hasData(e) || me.hasData(e)
		},
		data: function(e, t, n) {
			return ge.access(e, t, n)
		},
		removeData: function(e, t) {
			ge.remove(e, t)
		},
		_data: function(e, t, n) {
			return me.access(e, t, n)
		},
		_removeData: function(e, t) {
			me.remove(e, t)
		}
	}),
	Q.fn.extend({
		data: function(e, t) {
			var n, i, r, o = this[0],
			a = o && o.attributes;
			if (void 0 === e) {
				if (this.length && (r = ge.get(o), 1 === o.nodeType && !me.get(o, "hasDataAttrs"))) {
					for (n = a.length; n--;) a[n] && 0 === (i = a[n].name).indexOf("data-") && (i = Q.camelCase(i.slice(5)), l(o, i, r[i]));
					me.set(o, "hasDataAttrs", !0)
				}
				return r
			}
			return "object" == typeof e ? this.each(function() {
				ge.set(this, e)
			}) : pe(this,
			function(t) {
				var n, i = Q.camelCase(e);
				if (o && void 0 === t) {
					if (void 0 !== (n = ge.get(o, e))) return n;
					if (void 0 !== (n = ge.get(o, i))) return n;
					if (void 0 !== (n = l(o, i, void 0))) return n
				} else this.each(function() {
					var n = ge.get(this, i);
					ge.set(this, i, t),
					-1 !== e.indexOf("-") && void 0 !== n && ge.set(this, e, t)
				})
			},
			null, t, arguments.length > 1, null, !0)
		},
		removeData: function(e) {
			return this.each(function() {
				ge.remove(this, e)
			})
		}
	}),
	Q.extend({
		queue: function(e, t, n) {
			var i;
			return e ? (t = (t || "fx") + "queue", i = me.get(e, t), n && (!i || Q.isArray(n) ? i = me.access(e, t, Q.makeArray(n)) : i.push(n)), i || []) : void 0
		},
		dequeue: function(e, t) {
			t = t || "fx";
			var n = Q.queue(e, t),
			i = n.length,
			r = n.shift(),
			o = Q._queueHooks(e, t);
			"inprogress" === r && (r = n.shift(), i--),
			r && ("fx" === t && n.unshift("inprogress"), delete o.stop, r.call(e,
			function() {
				Q.dequeue(e, t)
			},
			o)),
			!i && o && o.empty.fire()
		},
		_queueHooks: function(e, t) {
			var n = t + "queueHooks";
			return me.get(e, n) || me.access(e, n, {
				empty: Q.Callbacks("once memory").add(function() {
					me.remove(e, [t + "queue", n])
				})
			})
		}
	}),
	Q.fn.extend({
		queue: function(e, t) {
			var n = 2;
			return "string" != typeof e && (t = e, e = "fx", n--),
			arguments.length < n ? Q.queue(this[0], e) : void 0 === t ? this: this.each(function() {
				var n = Q.queue(this, e, t);
				Q._queueHooks(this, e),
				"fx" === e && "inprogress" !== n[0] && Q.dequeue(this, e)
			})
		},
		dequeue: function(e) {
			return this.each(function() {
				Q.dequeue(this, e)
			})
		},
		clearQueue: function(e) {
			return this.queue(e || "fx", [])
		},
		promise: function(e, t) {
			var n, i = 1,
			r = Q.Deferred(),
			o = this,
			a = this.length,
			s = function() {--i || r.resolveWith(o, [o])
			};
			for ("string" != typeof e && (t = e, e = void 0), e = e || "fx"; a--;)(n = me.get(o[a], e + "queueHooks")) && n.empty && (i++, n.empty.add(s));
			return s(),
			r.promise(t)
		}
	});
	var be = /[+-]?(?:\d*\.|)\d+(?:[eE][+-]?\d+|)/.source,
	we = ["Top", "Right", "Bottom", "Left"],
	ke = function(e, t) {
		return e = t || e,
		"none" === Q.css(e, "display") || !Q.contains(e.ownerDocument, e)
	},
	xe = /^(?:checkbox|radio)$/i; !
	function() {
		var e = Z.createDocumentFragment().appendChild(Z.createElement("div")),
		t = Z.createElement("input");
		t.setAttribute("type", "radio"),
		t.setAttribute("checked", "checked"),
		t.setAttribute("name", "t"),
		e.appendChild(t),
		K.checkClone = e.cloneNode(!0).cloneNode(!0).lastChild.checked,
		e.innerHTML = "<textarea>x</textarea>",
		K.noCloneChecked = !!e.cloneNode(!0).lastChild.defaultValue
	} ();
	var _e = "undefined";
	K.focusinBubbles = "onfocusin" in e;
	var Ce = /^key/,
	Se = /^(?:mouse|pointer|contextmenu)|click/,
	Me = /^(?:focusinfocus|focusoutblur)$/,
	Te = /^([^.]*)(?:\.(.+)|)$/;
	Q.event = {
		global: {},
		add: function(e, t, n, i, r) {
			var o, a, s, l, c, u, d, h, f, p, m, g = me.get(e);
			if (g) for (n.handler && (o = n, n = o.handler, r = o.selector), n.guid || (n.guid = Q.guid++), (l = g.events) || (l = g.events = {}), (a = g.handle) || (a = g.handle = function(t) {
				return typeof Q !== _e && Q.event.triggered !== t.type ? Q.event.dispatch.apply(e, arguments) : void 0
			}), c = (t = (t || "").match(de) || [""]).length; c--;) s = Te.exec(t[c]) || [],
			f = m = s[1],
			p = (s[2] || "").split(".").sort(),
			f && (d = Q.event.special[f] || {},
			f = (r ? d.delegateType: d.bindType) || f, d = Q.event.special[f] || {},
			u = Q.extend({
				type: f,
				origType: m,
				data: i,
				handler: n,
				guid: n.guid,
				selector: r,
				needsContext: r && Q.expr.match.needsContext.test(r),
				namespace: p.join(".")
			},
			o), (h = l[f]) || (h = l[f] = [], h.delegateCount = 0, d.setup && !1 !== d.setup.call(e, i, p, a) || e.addEventListener && e.addEventListener(f, a, !1)), d.add && (d.add.call(e, u), u.handler.guid || (u.handler.guid = n.guid)), r ? h.splice(h.delegateCount++, 0, u) : h.push(u), Q.event.global[f] = !0)
		},
		remove: function(e, t, n, i, r) {
			var o, a, s, l, c, u, d, h, f, p, m, g = me.hasData(e) && me.get(e);
			if (g && (l = g.events)) {
				for (c = (t = (t || "").match(de) || [""]).length; c--;) if (s = Te.exec(t[c]) || [], f = m = s[1], p = (s[2] || "").split(".").sort(), f) {
					for (d = Q.event.special[f] || {},
					h = l[f = (i ? d.delegateType: d.bindType) || f] || [], s = s[2] && new RegExp("(^|\\.)" + p.join("\\.(?:.*\\.|)") + "(\\.|$)"), a = o = h.length; o--;) u = h[o],
					!r && m !== u.origType || n && n.guid !== u.guid || s && !s.test(u.namespace) || i && i !== u.selector && ("**" !== i || !u.selector) || (h.splice(o, 1), u.selector && h.delegateCount--, d.remove && d.remove.call(e, u));
					a && !h.length && (d.teardown && !1 !== d.teardown.call(e, p, g.handle) || Q.removeEvent(e, f, g.handle), delete l[f])
				} else for (f in l) Q.event.remove(e, f + t[c], n, i, !0);
				Q.isEmptyObject(l) && (delete g.handle, me.remove(e, "events"))
			}
		},
		trigger: function(t, n, i, r) {
			var o, a, s, l, c, u, d, h = [i || Z],
			f = V.call(t, "type") ? t.type: t,
			p = V.call(t, "namespace") ? t.namespace.split(".") : [];
			if (a = s = i = i || Z, 3 !== i.nodeType && 8 !== i.nodeType && !Me.test(f + Q.event.triggered) && (f.indexOf(".") >= 0 && (p = f.split("."), f = p.shift(), p.sort()), c = f.indexOf(":") < 0 && "on" + f, t = t[Q.expando] ? t: new Q.Event(f, "object" == typeof t && t), t.isTrigger = r ? 2 : 3, t.namespace = p.join("."), t.namespace_re = t.namespace ? new RegExp("(^|\\.)" + p.join("\\.(?:.*\\.|)") + "(\\.|$)") : null, t.result = void 0, t.target || (t.target = i), n = null == n ? [t] : Q.makeArray(n, [t]), d = Q.event.special[f] || {},
			r || !d.trigger || !1 !== d.trigger.apply(i, n))) {
				if (!r && !d.noBubble && !Q.isWindow(i)) {
					for (l = d.delegateType || f, Me.test(l + f) || (a = a.parentNode); a; a = a.parentNode) h.push(a),
					s = a;
					s === (i.ownerDocument || Z) && h.push(s.defaultView || s.parentWindow || e)
				}
				for (o = 0; (a = h[o++]) && !t.isPropagationStopped();) t.type = o > 1 ? l: d.bindType || f,
				(u = (me.get(a, "events") || {})[t.type] && me.get(a, "handle")) && u.apply(a, n),
				(u = c && a[c]) && u.apply && Q.acceptData(a) && (t.result = u.apply(a, n), !1 === t.result && t.preventDefault());
				return t.type = f,
				r || t.isDefaultPrevented() || d._default && !1 !== d._default.apply(h.pop(), n) || !Q.acceptData(i) || c && Q.isFunction(i[f]) && !Q.isWindow(i) && ((s = i[c]) && (i[c] = null), Q.event.triggered = f, i[f](), Q.event.triggered = void 0, s && (i[c] = s)),
				t.result
			}
		},
		dispatch: function(e) {
			e = Q.event.fix(e);
			var t, n, i, r, o, a = [],
			s = F.call(arguments),
			l = (me.get(this, "events") || {})[e.type] || [],
			c = Q.event.special[e.type] || {};
			if (s[0] = e, e.delegateTarget = this, !c.preDispatch || !1 !== c.preDispatch.call(this, e)) {
				for (a = Q.event.handlers.call(this, e, l), t = 0; (r = a[t++]) && !e.isPropagationStopped();) for (e.currentTarget = r.elem, n = 0; (o = r.handlers[n++]) && !e.isImmediatePropagationStopped();)(!e.namespace_re || e.namespace_re.test(o.namespace)) && (e.handleObj = o, e.data = o.data, void 0 !== (i = ((Q.event.special[o.origType] || {}).handle || o.handler).apply(r.elem, s)) && !1 === (e.result = i) && (e.preventDefault(), e.stopPropagation()));
				return c.postDispatch && c.postDispatch.call(this, e),
				e.result
			}
		},
		handlers: function(e, t) {
			var n, i, r, o, a = [],
			s = t.delegateCount,
			l = e.target;
			if (s && l.nodeType && (!e.button || "click" !== e.type)) for (; l !== this; l = l.parentNode || this) if (!0 !== l.disabled || "click" !== e.type) {
				for (i = [], n = 0; s > n; n++) o = t[n],
				r = o.selector + " ",
				void 0 === i[r] && (i[r] = o.needsContext ? Q(r, this).index(l) >= 0 : Q.find(r, this, null, [l]).length),
				i[r] && i.push(o);
				i.length && a.push({
					elem: l,
					handlers: i
				})
			}
			return s < t.length && a.push({
				elem: this,
				handlers: t.slice(s)
			}),
			a
		},
		props: "altKey bubbles cancelable ctrlKey currentTarget eventPhase metaKey relatedTarget shiftKey target timeStamp view which".split(" "),
		fixHooks: {},
		keyHooks: {
			props: "char charCode key keyCode".split(" "),
			filter: function(e, t) {
				return null == e.which && (e.which = null != t.charCode ? t.charCode: t.keyCode),
				e
			}
		},
		mouseHooks: {
			props: "button buttons clientX clientY offsetX offsetY pageX pageY screenX screenY toElement".split(" "),
			filter: function(e, t) {
				var n, i, r, o = t.button;
				return null == e.pageX && null != t.clientX && (n = e.target.ownerDocument || Z, i = n.documentElement, r = n.body, e.pageX = t.clientX + (i && i.scrollLeft || r && r.scrollLeft || 0) - (i && i.clientLeft || r && r.clientLeft || 0), e.pageY = t.clientY + (i && i.scrollTop || r && r.scrollTop || 0) - (i && i.clientTop || r && r.clientTop || 0)),
				e.which || void 0 === o || (e.which = 1 & o ? 1 : 2 & o ? 3 : 4 & o ? 2 : 0),
				e
			}
		},
		fix: function(e) {
			if (e[Q.expando]) return e;
			var t, n, i, r = e.type,
			o = e,
			a = this.fixHooks[r];
			for (a || (this.fixHooks[r] = a = Se.test(r) ? this.mouseHooks: Ce.test(r) ? this.keyHooks: {}), i = a.props ? this.props.concat(a.props) : this.props, e = new Q.Event(o), t = i.length; t--;) n = i[t],
			e[n] = o[n];
			return e.target || (e.target = Z),
			3 === e.target.nodeType && (e.target = e.target.parentNode),
			a.filter ? a.filter(e, o) : e
		},
		special: {
			load: {
				noBubble: !0
			},
			focus: {
				trigger: function() {
					return this !== d() && this.focus ? (this.focus(), !1) : void 0
				},
				delegateType: "focusin"
			},
			blur: {
				trigger: function() {
					return this === d() && this.blur ? (this.blur(), !1) : void 0
				},
				delegateType: "focusout"
			},
			click: {
				trigger: function() {
					return "checkbox" === this.type && this.click && Q.nodeName(this, "input") ? (this.click(), !1) : void 0
				},
				_default: function(e) {
					return Q.nodeName(e.target, "a")
				}
			},
			beforeunload: {
				postDispatch: function(e) {
					void 0 !== e.result && e.originalEvent && (e.originalEvent.returnValue = e.result)
				}
			}
		},
		simulate: function(e, t, n, i) {
			var r = Q.extend(new Q.Event, n, {
				type: e,
				isSimulated: !0,
				originalEvent: {}
			});
			i ? Q.event.trigger(r, null, t) : Q.event.dispatch.call(t, r),
			r.isDefaultPrevented() && n.preventDefault()
		}
	},
	Q.removeEvent = function(e, t, n) {
		e.removeEventListener && e.removeEventListener(t, n, !1)
	},
	Q.Event = function(e, t) {
		return this instanceof Q.Event ? (e && e.type ? (this.originalEvent = e, this.type = e.type, this.isDefaultPrevented = e.defaultPrevented || void 0 === e.defaultPrevented && !1 === e.returnValue ? c: u) : this.type = e, t && Q.extend(this, t), this.timeStamp = e && e.timeStamp || Q.now(), void(this[Q.expando] = !0)) : new Q.Event(e, t)
	},
	Q.Event.prototype = {
		isDefaultPrevented: u,
		isPropagationStopped: u,
		isImmediatePropagationStopped: u,
		preventDefault: function() {
			var e = this.originalEvent;
			this.isDefaultPrevented = c,
			e && e.preventDefault && e.preventDefault()
		},
		stopPropagation: function() {
			var e = this.originalEvent;
			this.isPropagationStopped = c,
			e && e.stopPropagation && e.stopPropagation()
		},
		stopImmediatePropagation: function() {
			var e = this.originalEvent;
			this.isImmediatePropagationStopped = c,
			e && e.stopImmediatePropagation && e.stopImmediatePropagation(),
			this.stopPropagation()
		}
	},
	Q.each({
		mouseenter: "mouseover",
		mouseleave: "mouseout",
		pointerenter: "pointerover",
		pointerleave: "pointerout"
	},
	function(e, t) {
		Q.event.special[e] = {
			delegateType: t,
			bindType: t,
			handle: function(e) {
				var n, i = this,
				r = e.relatedTarget,
				o = e.handleObj;
				return (!r || r !== i && !Q.contains(i, r)) && (e.type = o.origType, n = o.handler.apply(this, arguments), e.type = t),
				n
			}
		}
	}),
	K.focusinBubbles || Q.each({
		focus: "focusin",
		blur: "focusout"
	},
	function(e, t) {
		var n = function(e) {
			Q.event.simulate(t, e.target, Q.event.fix(e), !0)
		};
		Q.event.special[t] = {
			setup: function() {
				var i = this.ownerDocument || this,
				r = me.access(i, t);
				r || i.addEventListener(e, n, !0),
				me.access(i, t, (r || 0) + 1)
			},
			teardown: function() {
				var i = this.ownerDocument || this,
				r = me.access(i, t) - 1;
				r ? me.access(i, t, r) : (i.removeEventListener(e, n, !0), me.remove(i, t))
			}
		}
	}),
	Q.fn.extend({
		on: function(e, t, n, i, r) {
			var o, a;
			if ("object" == typeof e) {
				"string" != typeof t && (n = n || t, t = void 0);
				for (a in e) this.on(a, t, n, e[a], r);
				return this
			}
			if (null == n && null == i ? (i = t, n = t = void 0) : null == i && ("string" == typeof t ? (i = n, n = void 0) : (i = n, n = t, t = void 0)), !1 === i) i = u;
			else if (!i) return this;
			return 1 === r && (o = i, i = function(e) {
				return Q().off(e),
				o.apply(this, arguments)
			},
			i.guid = o.guid || (o.guid = Q.guid++)),
			this.each(function() {
				Q.event.add(this, e, i, n, t)
			})
		},
		one: function(e, t, n, i) {
			return this.on(e, t, n, i, 1)
		},
		off: function(e, t, n) {
			var i, r;
			if (e && e.preventDefault && e.handleObj) return i = e.handleObj,
			Q(e.delegateTarget).off(i.namespace ? i.origType + "." + i.namespace: i.origType, i.selector, i.handler),
			this;
			if ("object" == typeof e) {
				for (r in e) this.off(r, t, e[r]);
				return this
			}
			return (!1 === t || "function" == typeof t) && (n = t, t = void 0),
			!1 === n && (n = u),
			this.each(function() {
				Q.event.remove(this, e, n, t)
			})
		},
		trigger: function(e, t) {
			return this.each(function() {
				Q.event.trigger(e, t, this)
			})
		},
		triggerHandler: function(e, t) {
			var n = this[0];
			return n ? Q.event.trigger(e, t, n, !0) : void 0
		}
	});
	var De = /<(?!area|br|col|embed|hr|img|input|link|meta|param)(([\w:]+)[^>]*)\/>/gi,
	Le = /<([\w:]+)/,
	Oe = /<|&#?\w+;/,
	Ne = /<(?:script|style|link)/i,
	Ae = /checked\s*(?:[^=]|=\s*.checked.)/i,
	Ee = /^$|\/(?:java|ecma)script/i,
	$e = /^true\/(.*)/,
	qe = /^\s*<!(?:\[CDATA\[|--)|(?:\]\]|--)>\s*$/g,
	je = {
		option: [1, "<select multiple='multiple'>", "</select>"],
		thead: [1, "<table>", "</table>"],
		col: [2, "<table><colgroup>", "</colgroup></table>"],
		tr: [2, "<table><tbody>", "</tbody></table>"],
		td: [3, "<table><tbody><tr>", "</tr></tbody></table>"],
		_default: [0, "", ""]
	};
	je.optgroup = je.option,
	je.tbody = je.tfoot = je.colgroup = je.caption = je.thead,
	je.th = je.td,
	Q.extend({
		clone: function(e, t, n) {
			var i, r, o, a, s = e.cloneNode(!0),
			l = Q.contains(e.ownerDocument, e);
			if (! (K.noCloneChecked || 1 !== e.nodeType && 11 !== e.nodeType || Q.isXMLDoc(e))) for (a = v(s), o = v(e), i = 0, r = o.length; r > i; i++) y(o[i], a[i]);
			if (t) if (n) for (o = o || v(e), a = a || v(s), i = 0, r = o.length; r > i; i++) g(o[i], a[i]);
			else g(e, s);
			return (a = v(s, "script")).length > 0 && m(a, !l && v(e, "script")),
			s
		},
		buildFragment: function(e, t, n, i) {
			for (var r, o, a, s, l, c, u = t.createDocumentFragment(), d = [], h = 0, f = e.length; f > h; h++) if ((r = e[h]) || 0 === r) if ("object" === Q.type(r)) Q.merge(d, r.nodeType ? [r] : r);
			else if (Oe.test(r)) {
				for (o = o || u.appendChild(t.createElement("div")), a = (Le.exec(r) || ["", ""])[1].toLowerCase(), s = je[a] || je._default, o.innerHTML = s[1] + r.replace(De, "<$1></$2>") + s[2], c = s[0]; c--;) o = o.lastChild;
				Q.merge(d, o.childNodes),
				(o = u.firstChild).textContent = ""
			} else d.push(t.createTextNode(r));
			for (u.textContent = "", h = 0; r = d[h++];) if ((!i || -1 === Q.inArray(r, i)) && (l = Q.contains(r.ownerDocument, r), o = v(u.appendChild(r), "script"), l && m(o), n)) for (c = 0; r = o[c++];) Ee.test(r.type || "") && n.push(r);
			return u
		},
		cleanData: function(e) {
			for (var t, n, i, r, o = Q.event.special,
			a = 0; void 0 !== (n = e[a]); a++) {
				if (Q.acceptData(n) && (r = n[me.expando]) && (t = me.cache[r])) {
					if (t.events) for (i in t.events) o[i] ? Q.event.remove(n, i) : Q.removeEvent(n, i, t.handle);
					me.cache[r] && delete me.cache[r]
				}
				delete ge.cache[n[ge.expando]]
			}
		}
	}),
	Q.fn.extend({
		text: function(e) {
			return pe(this,
			function(e) {
				return void 0 === e ? Q.text(this) : this.empty().each(function() { (1 === this.nodeType || 11 === this.nodeType || 9 === this.nodeType) && (this.textContent = e)
				})
			},
			null, e, arguments.length)
		},
		append: function() {
			return this.domManip(arguments,
			function(e) {
				1 !== this.nodeType && 11 !== this.nodeType && 9 !== this.nodeType || h(this, e).appendChild(e)
			})
		},
		prepend: function() {
			return this.domManip(arguments,
			function(e) {
				if (1 === this.nodeType || 11 === this.nodeType || 9 === this.nodeType) {
					var t = h(this, e);
					t.insertBefore(e, t.firstChild)
				}
			})
		},
		before: function() {
			return this.domManip(arguments,
			function(e) {
				this.parentNode && this.parentNode.insertBefore(e, this)
			})
		},
		after: function() {
			return this.domManip(arguments,
			function(e) {
				this.parentNode && this.parentNode.insertBefore(e, this.nextSibling)
			})
		},
		remove: function(e, t) {
			for (var n, i = e ? Q.filter(e, this) : this, r = 0; null != (n = i[r]); r++) t || 1 !== n.nodeType || Q.cleanData(v(n)),
			n.parentNode && (t && Q.contains(n.ownerDocument, n) && m(v(n, "script")), n.parentNode.removeChild(n));
			return this
		},
		empty: function() {
			for (var e, t = 0; null != (e = this[t]); t++) 1 === e.nodeType && (Q.cleanData(v(e, !1)), e.textContent = "");
			return this
		},
		clone: function(e, t) {
			return e = null != e && e,
			t = null == t ? e: t,
			this.map(function() {
				return Q.clone(this, e, t)
			})
		},
		html: function(e) {
			return pe(this,
			function(e) {
				var t = this[0] || {},
				n = 0,
				i = this.length;
				if (void 0 === e && 1 === t.nodeType) return t.innerHTML;
				if ("string" == typeof e && !Ne.test(e) && !je[(Le.exec(e) || ["", ""])[1].toLowerCase()]) {
					e = e.replace(De, "<$1></$2>");
					try {
						for (; i > n; n++) 1 === (t = this[n] || {}).nodeType && (Q.cleanData(v(t, !1)), t.innerHTML = e);
						t = 0
					} catch(e) {}
				}
				t && this.empty().append(e)
			},
			null, e, arguments.length)
		},
		replaceWith: function() {
			var e = arguments[0];
			return this.domManip(arguments,
			function(t) {
				e = this.parentNode,
				Q.cleanData(v(this)),
				e && e.replaceChild(t, this)
			}),
			e && (e.length || e.nodeType) ? this: this.remove()
		},
		detach: function(e) {
			return this.remove(e, !0)
		},
		domManip: function(e, t) {
			e = Y.apply([], e);
			var n, i, r, o, a, s, l = 0,
			c = this.length,
			u = this,
			d = c - 1,
			h = e[0],
			m = Q.isFunction(h);
			if (m || c > 1 && "string" == typeof h && !K.checkClone && Ae.test(h)) return this.each(function(n) {
				var i = u.eq(n);
				m && (e[0] = h.call(this, n, i.html())),
				i.domManip(e, t)
			});
			if (c && (n = Q.buildFragment(e, this[0].ownerDocument, !1, this), i = n.firstChild, 1 === n.childNodes.length && (n = i), i)) {
				for (o = (r = Q.map(v(n, "script"), f)).length; c > l; l++) a = n,
				l !== d && (a = Q.clone(a, !0, !0), o && Q.merge(r, v(a, "script"))),
				t.call(this[l], a, l);
				if (o) for (s = r[r.length - 1].ownerDocument, Q.map(r, p), l = 0; o > l; l++) a = r[l],
				Ee.test(a.type || "") && !me.access(a, "globalEval") && Q.contains(s, a) && (a.src ? Q._evalUrl && Q._evalUrl(a.src) : Q.globalEval(a.textContent.replace(qe, "")))
			}
			return this
		}
	}),
	Q.each({
		appendTo: "append",
		prependTo: "prepend",
		insertBefore: "before",
		insertAfter: "after",
		replaceAll: "replaceWith"
	},
	function(e, t) {
		Q.fn[e] = function(e) {
			for (var n, i = [], r = Q(e), o = r.length - 1, a = 0; o >= a; a++) n = a === o ? this: this.clone(!0),
			Q(r[a])[t](n),
			R.apply(i, n.get());
			return this.pushStack(i)
		}
	});
	var Pe, Ie = {},
	ze = /^margin/,
	We = new RegExp("^(" + be + ")(?!px)[a-z%]+$", "i"),
	He = function(t) {
		return t.ownerDocument.defaultView.opener ? t.ownerDocument.defaultView.getComputedStyle(t, null) : e.getComputedStyle(t, null)
	}; !
	function() {
		function t() {
			a.style.cssText = "-webkit-box-sizing:border-box;-moz-box-sizing:border-box;box-sizing:border-box;display:block;margin-top:1%;top:1%;border:1px;padding:1px;width:4px;position:absolute",
			a.innerHTML = "",
			r.appendChild(o);
			var t = e.getComputedStyle(a, null);
			n = "1%" !== t.top,
			i = "4px" === t.width,
			r.removeChild(o)
		}
		var n, i, r = Z.documentElement,
		o = Z.createElement("div"),
		a = Z.createElement("div");
		a.style && (a.style.backgroundClip = "content-box", a.cloneNode(!0).style.backgroundClip = "", K.clearCloneStyle = "content-box" === a.style.backgroundClip, o.style.cssText = "border:0;width:0;height:0;top:0;left:-9999px;margin-top:1px;position:absolute", o.appendChild(a), e.getComputedStyle && Q.extend(K, {
			pixelPosition: function() {
				return t(),
				n
			},
			boxSizingReliable: function() {
				return null == i && t(),
				i
			},
			reliableMarginRight: function() {
				var t, n = a.appendChild(Z.createElement("div"));
				return n.style.cssText = a.style.cssText = "-webkit-box-sizing:content-box;-moz-box-sizing:content-box;box-sizing:content-box;display:block;margin:0;border:0;padding:0",
				n.style.marginRight = n.style.width = "0",
				a.style.width = "1px",
				r.appendChild(o),
				t = !parseFloat(e.getComputedStyle(n, null).marginRight),
				r.removeChild(o),
				a.removeChild(n),
				t
			}
		}))
	} (),
	Q.swap = function(e, t, n, i) {
		var r, o, a = {};
		for (o in t) a[o] = e.style[o],
		e.style[o] = t[o];
		r = n.apply(e, i || []);
		for (o in t) e.style[o] = a[o];
		return r
	};
	var Fe = /^(none|table(?!-c[ea]).+)/,
	Ye = new RegExp("^(" + be + ")(.*)$", "i"),
	Re = new RegExp("^([+-])=(" + be + ")", "i"),
	Ue = {
		position: "absolute",
		visibility: "hidden",
		display: "block"
	},
	Be = {
		letterSpacing: "0",
		fontWeight: "400"
	},
	Ge = ["Webkit", "O", "Moz", "ms"];
	Q.extend({
		cssHooks: {
			opacity: {
				get: function(e, t) {
					if (t) {
						var n = k(e, "opacity");
						return "" === n ? "1": n
					}
				}
			}
		},
		cssNumber: {
			columnCount: !0,
			fillOpacity: !0,
			flexGrow: !0,
			flexShrink: !0,
			fontWeight: !0,
			lineHeight: !0,
			opacity: !0,
			order: !0,
			orphans: !0,
			widows: !0,
			zIndex: !0,
			zoom: !0
		},
		cssProps: {
			float: "cssFloat"
		},
		style: function(e, t, n, i) {
			if (e && 3 !== e.nodeType && 8 !== e.nodeType && e.style) {
				var r, o, a, s = Q.camelCase(t),
				l = e.style;
				return t = Q.cssProps[s] || (Q.cssProps[s] = _(l, s)),
				a = Q.cssHooks[t] || Q.cssHooks[s],
				void 0 === n ? a && "get" in a && void 0 !== (r = a.get(e, !1, i)) ? r: l[t] : ("string" === (o = typeof n) && (r = Re.exec(n)) && (n = (r[1] + 1) * r[2] + parseFloat(Q.css(e, t)), o = "number"), void(null != n && n === n && ("number" !== o || Q.cssNumber[s] || (n += "px"), K.clearCloneStyle || "" !== n || 0 !== t.indexOf("background") || (l[t] = "inherit"), a && "set" in a && void 0 === (n = a.set(e, n, i)) || (l[t] = n))))
			}
		},
		css: function(e, t, n, i) {
			var r, o, a, s = Q.camelCase(t);
			return t = Q.cssProps[s] || (Q.cssProps[s] = _(e.style, s)),
			(a = Q.cssHooks[t] || Q.cssHooks[s]) && "get" in a && (r = a.get(e, !0, n)),
			void 0 === r && (r = k(e, t, i)),
			"normal" === r && t in Be && (r = Be[t]),
			"" === n || n ? (o = parseFloat(r), !0 === n || Q.isNumeric(o) ? o || 0 : r) : r
		}
	}),
	Q.each(["height", "width"],
	function(e, t) {
		Q.cssHooks[t] = {
			get: function(e, n, i) {
				return n ? Fe.test(Q.css(e, "display")) && 0 === e.offsetWidth ? Q.swap(e, Ue,
				function() {
					return M(e, t, i)
				}) : M(e, t, i) : void 0
			},
			set: function(e, n, i) {
				var r = i && He(e);
				return C(0, n, i ? S(e, t, i, "border-box" === Q.css(e, "boxSizing", !1, r), r) : 0)
			}
		}
	}),
	Q.cssHooks.marginRight = x(K.reliableMarginRight,
	function(e, t) {
		return t ? Q.swap(e, {
			display: "inline-block"
		},
		k, [e, "marginRight"]) : void 0
	}),
	Q.each({
		margin: "",
		padding: "",
		border: "Width"
	},
	function(e, t) {
		Q.cssHooks[e + t] = {
			expand: function(n) {
				for (var i = 0,
				r = {},
				o = "string" == typeof n ? n.split(" ") : [n]; 4 > i; i++) r[e + we[i] + t] = o[i] || o[i - 2] || o[0];
				return r
			}
		},
		ze.test(e) || (Q.cssHooks[e + t].set = C)
	}),
	Q.fn.extend({
		css: function(e, t) {
			return pe(this,
			function(e, t, n) {
				var i, r, o = {},
				a = 0;
				if (Q.isArray(t)) {
					for (i = He(e), r = t.length; r > a; a++) o[t[a]] = Q.css(e, t[a], !1, i);
					return o
				}
				return void 0 !== n ? Q.style(e, t, n) : Q.css(e, t)
			},
			e, t, arguments.length > 1)
		},
		show: function() {
			return T(this, !0)
		},
		hide: function() {
			return T(this)
		},
		toggle: function(e) {
			return "boolean" == typeof e ? e ? this.show() : this.hide() : this.each(function() {
				ke(this) ? Q(this).show() : Q(this).hide()
			})
		}
	}),
	Q.Tween = D,
	D.prototype = {
		constructor: D,
		init: function(e, t, n, i, r, o) {
			this.elem = e,
			this.prop = n,
			this.easing = r || "swing",
			this.options = t,
			this.start = this.now = this.cur(),
			this.end = i,
			this.unit = o || (Q.cssNumber[n] ? "": "px")
		},
		cur: function() {
			var e = D.propHooks[this.prop];
			return e && e.get ? e.get(this) : D.propHooks._default.get(this)
		},
		run: function(e) {
			var t, n = D.propHooks[this.prop];
			return this.options.duration ? this.pos = t = Q.easing[this.easing](e, this.options.duration * e, 0, 1, this.options.duration) : this.pos = t = e,
			this.now = (this.end - this.start) * t + this.start,
			this.options.step && this.options.step.call(this.elem, this.now, this),
			n && n.set ? n.set(this) : D.propHooks._default.set(this),
			this
		}
	},
	D.prototype.init.prototype = D.prototype,
	D.propHooks = {
		_default: {
			get: function(e) {
				var t;
				return null == e.elem[e.prop] || e.elem.style && null != e.elem.style[e.prop] ? (t = Q.css(e.elem, e.prop, "")) && "auto" !== t ? t: 0 : e.elem[e.prop]
			},
			set: function(e) {
				Q.fx.step[e.prop] ? Q.fx.step[e.prop](e) : e.elem.style && (null != e.elem.style[Q.cssProps[e.prop]] || Q.cssHooks[e.prop]) ? Q.style(e.elem, e.prop, e.now + e.unit) : e.elem[e.prop] = e.now
			}
		}
	},
	D.propHooks.scrollTop = D.propHooks.scrollLeft = {
		set: function(e) {
			e.elem.nodeType && e.elem.parentNode && (e.elem[e.prop] = e.now)
		}
	},
	Q.easing = {
		linear: function(e) {
			return e
		},
		swing: function(e) {
			return.5 - Math.cos(e * Math.PI) / 2
		}
	},
	Q.fx = D.prototype.init,
	Q.fx.step = {};
	var Ve, Ke, Ze = /^(?:toggle|show|hide)$/,
	Xe = new RegExp("^(?:([+-])=|)(" + be + ")([a-z%]*)$", "i"),
	Qe = /queueHooks$/,
	Je = [function(e, t, n) {
		var i, r, o, a, s, l, c, u = this,
		d = {},
		h = e.style,
		f = e.nodeType && ke(e),
		p = me.get(e, "fxshow");
		n.queue || (null == (s = Q._queueHooks(e, "fx")).unqueued && (s.unqueued = 0, l = s.empty.fire, s.empty.fire = function() {
			s.unqueued || l()
		}), s.unqueued++, u.always(function() {
			u.always(function() {
				s.unqueued--,
				Q.queue(e, "fx").length || s.empty.fire()
			})
		})),
		1 === e.nodeType && ("height" in t || "width" in t) && (n.overflow = [h.overflow, h.overflowX, h.overflowY], "inline" === ("none" === (c = Q.css(e, "display")) ? me.get(e, "olddisplay") || w(e.nodeName) : c) && "none" === Q.css(e, "float") && (h.display = "inline-block")),
		n.overflow && (h.overflow = "hidden", u.always(function() {
			h.overflow = n.overflow[0],
			h.overflowX = n.overflow[1],
			h.overflowY = n.overflow[2]
		}));
		for (i in t) if (r = t[i], Ze.exec(r)) {
			if (delete t[i], o = o || "toggle" === r, r === (f ? "hide": "show")) {
				if ("show" !== r || !p || void 0 === p[i]) continue;
				f = !0
			}
			d[i] = p && p[i] || Q.style(e, i)
		} else c = void 0;
		if (Q.isEmptyObject(d))"inline" === ("none" === c ? w(e.nodeName) : c) && (h.display = c);
		else {
			p ? "hidden" in p && (f = p.hidden) : p = me.access(e, "fxshow", {}),
			o && (p.hidden = !f),
			f ? Q(e).show() : u.done(function() {
				Q(e).hide()
			}),
			u.done(function() {
				var t;
				me.remove(e, "fxshow");
				for (t in d) Q.style(e, t, d[t])
			});
			for (i in d) a = N(f ? p[i] : 0, i, u),
			i in p || (p[i] = a.start, f && (a.end = a.start, a.start = "width" === i || "height" === i ? 1 : 0))
		}
	}],
	et = {
		"*": [function(e, t) {
			var n = this.createTween(e, t),
			i = n.cur(),
			r = Xe.exec(t),
			o = r && r[3] || (Q.cssNumber[e] ? "": "px"),
			a = (Q.cssNumber[e] || "px" !== o && +i) && Xe.exec(Q.css(n.elem, e)),
			s = 1,
			l = 20;
			if (a && a[3] !== o) {
				o = o || a[3],
				r = r || [],
				a = +i || 1;
				do {
					s = s || ".5", a /= s, Q.style(n.elem, e, a + o)
				} while ( s !== ( s = n . cur () / i) && 1 !== s && --l)
			}
			return r && (a = n.start = +a || +i || 0, n.unit = o, n.end = r[1] ? a + (r[1] + 1) * r[2] : +r[2]),
			n
		}]
	};
	Q.Animation = Q.extend(E, {
		tweener: function(e, t) {
			Q.isFunction(e) ? (t = e, e = ["*"]) : e = e.split(" ");
			for (var n, i = 0,
			r = e.length; r > i; i++) n = e[i],
			et[n] = et[n] || [],
			et[n].unshift(t)
		},
		prefilter: function(e, t) {
			t ? Je.unshift(e) : Je.push(e)
		}
	}),
	Q.speed = function(e, t, n) {
		var i = e && "object" == typeof e ? Q.extend({},
		e) : {
			complete: n || !n && t || Q.isFunction(e) && e,
			duration: e,
			easing: n && t || t && !Q.isFunction(t) && t
		};
		return i.duration = Q.fx.off ? 0 : "number" == typeof i.duration ? i.duration: i.duration in Q.fx.speeds ? Q.fx.speeds[i.duration] : Q.fx.speeds._default,
		(null == i.queue || !0 === i.queue) && (i.queue = "fx"),
		i.old = i.complete,
		i.complete = function() {
			Q.isFunction(i.old) && i.old.call(this),
			i.queue && Q.dequeue(this, i.queue)
		},
		i
	},
	Q.fn.extend({
		fadeTo: function(e, t, n, i) {
			return this.filter(ke).css("opacity", 0).show().end().animate({
				opacity: t
			},
			e, n, i)
		},
		animate: function(e, t, n, i) {
			var r = Q.isEmptyObject(e),
			o = Q.speed(t, n, i),
			a = function() {
				var t = E(this, Q.extend({},
				e), o); (r || me.get(this, "finish")) && t.stop(!0)
			};
			return a.finish = a,
			r || !1 === o.queue ? this.each(a) : this.queue(o.queue, a)
		},
		stop: function(e, t, n) {
			var i = function(e) {
				var t = e.stop;
				delete e.stop,
				t(n)
			};
			return "string" != typeof e && (n = t, t = e, e = void 0),
			t && !1 !== e && this.queue(e || "fx", []),
			this.each(function() {
				var t = !0,
				r = null != e && e + "queueHooks",
				o = Q.timers,
				a = me.get(this);
				if (r) a[r] && a[r].stop && i(a[r]);
				else for (r in a) a[r] && a[r].stop && Qe.test(r) && i(a[r]);
				for (r = o.length; r--;) o[r].elem !== this || null != e && o[r].queue !== e || (o[r].anim.stop(n), t = !1, o.splice(r, 1)); (t || !n) && Q.dequeue(this, e)
			})
		},
		finish: function(e) {
			return ! 1 !== e && (e = e || "fx"),
			this.each(function() {
				var t, n = me.get(this),
				i = n[e + "queue"],
				r = n[e + "queueHooks"],
				o = Q.timers,
				a = i ? i.length: 0;
				for (n.finish = !0, Q.queue(this, e, []), r && r.stop && r.stop.call(this, !0), t = o.length; t--;) o[t].elem === this && o[t].queue === e && (o[t].anim.stop(!0), o.splice(t, 1));
				for (t = 0; a > t; t++) i[t] && i[t].finish && i[t].finish.call(this);
				delete n.finish
			})
		}
	}),
	Q.each(["toggle", "show", "hide"],
	function(e, t) {
		var n = Q.fn[t];
		Q.fn[t] = function(e, i, r) {
			return null == e || "boolean" == typeof e ? n.apply(this, arguments) : this.animate(O(t, !0), e, i, r)
		}
	}),
	Q.each({
		slideDown: O("show"),
		slideUp: O("hide"),
		slideToggle: O("toggle"),
		fadeIn: {
			opacity: "show"
		},
		fadeOut: {
			opacity: "hide"
		},
		fadeToggle: {
			opacity: "toggle"
		}
	},
	function(e, t) {
		Q.fn[e] = function(e, n, i) {
			return this.animate(t, e, n, i)
		}
	}),
	Q.timers = [],
	Q.fx.tick = function() {
		var e, t = 0,
		n = Q.timers;
		for (Ve = Q.now(); t < n.length; t++)(e = n[t])() || n[t] !== e || n.splice(t--, 1);
		n.length || Q.fx.stop(),
		Ve = void 0
	},
	Q.fx.timer = function(e) {
		Q.timers.push(e),
		e() ? Q.fx.start() : Q.timers.pop()
	},
	Q.fx.interval = 13,
	Q.fx.start = function() {
		Ke || (Ke = setInterval(Q.fx.tick, Q.fx.interval))
	},
	Q.fx.stop = function() {
		clearInterval(Ke),
		Ke = null
	},
	Q.fx.speeds = {
		slow: 600,
		fast: 200,
		_default: 400
	},
	Q.fn.delay = function(e, t) {
		return e = Q.fx ? Q.fx.speeds[e] || e: e,
		t = t || "fx",
		this.queue(t,
		function(t, n) {
			var i = setTimeout(t, e);
			n.stop = function() {
				clearTimeout(i)
			}
		})
	},
	function() {
		var e = Z.createElement("input"),
		t = Z.createElement("select"),
		n = t.appendChild(Z.createElement("option"));
		e.type = "checkbox",
		K.checkOn = "" !== e.value,
		K.optSelected = n.selected,
		t.disabled = !0,
		K.optDisabled = !n.disabled,
		(e = Z.createElement("input")).value = "t",
		e.type = "radio",
		K.radioValue = "t" === e.value
	} ();
	var tt, nt = Q.expr.attrHandle;
	Q.fn.extend({
		attr: function(e, t) {
			return pe(this, Q.attr, e, t, arguments.length > 1)
		},
		removeAttr: function(e) {
			return this.each(function() {
				Q.removeAttr(this, e)
			})
		}
	}),
	Q.extend({
		attr: function(e, t, n) {
			var i, r, o = e.nodeType;
			if (e && 3 !== o && 8 !== o && 2 !== o) return typeof e.getAttribute === _e ? Q.prop(e, t, n) : (1 === o && Q.isXMLDoc(e) || (t = t.toLowerCase(), i = Q.attrHooks[t] || (Q.expr.match.bool.test(t) ? tt: void 0)), void 0 === n ? i && "get" in i && null !== (r = i.get(e, t)) ? r: null == (r = Q.find.attr(e, t)) ? void 0 : r: null !== n ? i && "set" in i && void 0 !== (r = i.set(e, n, t)) ? r: (e.setAttribute(t, n + ""), n) : void Q.removeAttr(e, t))
		},
		removeAttr: function(e, t) {
			var n, i, r = 0,
			o = t && t.match(de);
			if (o && 1 === e.nodeType) for (; n = o[r++];) i = Q.propFix[n] || n,
			Q.expr.match.bool.test(n) && (e[i] = !1),
			e.removeAttribute(n)
		},
		attrHooks: {
			type: {
				set: function(e, t) {
					if (!K.radioValue && "radio" === t && Q.nodeName(e, "input")) {
						var n = e.value;
						return e.setAttribute("type", t),
						n && (e.value = n),
						t
					}
				}
			}
		}
	}),
	tt = {
		set: function(e, t, n) {
			return ! 1 === t ? Q.removeAttr(e, n) : e.setAttribute(n, n),
			n
		}
	},
	Q.each(Q.expr.match.bool.source.match(/\w+/g),
	function(e, t) {
		var n = nt[t] || Q.find.attr;
		nt[t] = function(e, t, i) {
			var r, o;
			return i || (o = nt[t], nt[t] = r, r = null != n(e, t, i) ? t.toLowerCase() : null, nt[t] = o),
			r
		}
	});
	var it = /^(?:input|select|textarea|button)$/i;
	Q.fn.extend({
		prop: function(e, t) {
			return pe(this, Q.prop, e, t, arguments.length > 1)
		},
		removeProp: function(e) {
			return this.each(function() {
				delete this[Q.propFix[e] || e]
			})
		}
	}),
	Q.extend({
		propFix: {
			for: "htmlFor",
			class: "className"
		},
		prop: function(e, t, n) {
			var i, r, o = e.nodeType;
			if (e && 3 !== o && 8 !== o && 2 !== o) return (1 !== o || !Q.isXMLDoc(e)) && (t = Q.propFix[t] || t, r = Q.propHooks[t]),
			void 0 !== n ? r && "set" in r && void 0 !== (i = r.set(e, n, t)) ? i: e[t] = n: r && "get" in r && null !== (i = r.get(e, t)) ? i: e[t]
		},
		propHooks: {
			tabIndex: {
				get: function(e) {
					return e.hasAttribute("tabindex") || it.test(e.nodeName) || e.href ? e.tabIndex: -1
				}
			}
		}
	}),
	K.optSelected || (Q.propHooks.selected = {
		get: function(e) {
			var t = e.parentNode;
			return t && t.parentNode && t.parentNode.selectedIndex,
			null
		}
	}),
	Q.each(["tabIndex", "readOnly", "maxLength", "cellSpacing", "cellPadding", "rowSpan", "colSpan", "useMap", "frameBorder", "contentEditable"],
	function() {
		Q.propFix[this.toLowerCase()] = this
	});
	var rt = /[\t\r\n\f]/g;
	Q.fn.extend({
		addClass: function(e) {
			var t, n, i, r, o, a, s = "string" == typeof e && e,
			l = 0,
			c = this.length;
			if (Q.isFunction(e)) return this.each(function(t) {
				Q(this).addClass(e.call(this, t, this.className))
			});
			if (s) for (t = (e || "").match(de) || []; c > l; l++) if (n = this[l], i = 1 === n.nodeType && (n.className ? (" " + n.className + " ").replace(rt, " ") : " ")) {
				for (o = 0; r = t[o++];) i.indexOf(" " + r + " ") < 0 && (i += r + " ");
				a = Q.trim(i),
				n.className !== a && (n.className = a)
			}
			return this
		},
		removeClass: function(e) {
			var t, n, i, r, o, a, s = 0 === arguments.length || "string" == typeof e && e,
			l = 0,
			c = this.length;
			if (Q.isFunction(e)) return this.each(function(t) {
				Q(this).removeClass(e.call(this, t, this.className))
			});
			if (s) for (t = (e || "").match(de) || []; c > l; l++) if (n = this[l], i = 1 === n.nodeType && (n.className ? (" " + n.className + " ").replace(rt, " ") : "")) {
				for (o = 0; r = t[o++];) for (; i.indexOf(" " + r + " ") >= 0;) i = i.replace(" " + r + " ", " ");
				a = e ? Q.trim(i) : "",
				n.className !== a && (n.className = a)
			}
			return this
		},
		toggleClass: function(e, t) {
			var n = typeof e;
			return "boolean" == typeof t && "string" === n ? t ? this.addClass(e) : this.removeClass(e) : this.each(Q.isFunction(e) ?
			function(n) {
				Q(this).toggleClass(e.call(this, n, this.className, t), t)
			}: function() {
				if ("string" === n) for (var t, i = 0,
				r = Q(this), o = e.match(de) || []; t = o[i++];) r.hasClass(t) ? r.removeClass(t) : r.addClass(t);
				else(n === _e || "boolean" === n) && (this.className && me.set(this, "__className__", this.className), this.className = this.className || !1 === e ? "": me.get(this, "__className__") || "")
			})
		},
		hasClass: function(e) {
			for (var t = " " + e + " ",
			n = 0,
			i = this.length; i > n; n++) if (1 === this[n].nodeType && (" " + this[n].className + " ").replace(rt, " ").indexOf(t) >= 0) return ! 0;
			return ! 1
		}
	});
	var ot = /\r/g;
	Q.fn.extend({
		val: function(e) {
			var t, n, i, r = this[0];
			return arguments.length ? (i = Q.isFunction(e), this.each(function(n) {
				var r;
				1 === this.nodeType && (null == (r = i ? e.call(this, n, Q(this).val()) : e) ? r = "": "number" == typeof r ? r += "": Q.isArray(r) && (r = Q.map(r,
				function(e) {
					return null == e ? "": e + ""
				})), (t = Q.valHooks[this.type] || Q.valHooks[this.nodeName.toLowerCase()]) && "set" in t && void 0 !== t.set(this, r, "value") || (this.value = r))
			})) : r ? (t = Q.valHooks[r.type] || Q.valHooks[r.nodeName.toLowerCase()]) && "get" in t && void 0 !== (n = t.get(r, "value")) ? n: "string" == typeof(n = r.value) ? n.replace(ot, "") : null == n ? "": n: void 0
		}
	}),
	Q.extend({
		valHooks: {
			option: {
				get: function(e) {
					var t = Q.find.attr(e, "value");
					return null != t ? t: Q.trim(Q.text(e))
				}
			},
			select: {
				get: function(e) {
					for (var t, n, i = e.options,
					r = e.selectedIndex,
					o = "select-one" === e.type || 0 > r,
					a = o ? null: [], s = o ? r + 1 : i.length, l = 0 > r ? s: o ? r: 0; s > l; l++) if (! (! (n = i[l]).selected && l !== r || (K.optDisabled ? n.disabled: null !== n.getAttribute("disabled")) || n.parentNode.disabled && Q.nodeName(n.parentNode, "optgroup"))) {
						if (t = Q(n).val(), o) return t;
						a.push(t)
					}
					return a
				},
				set: function(e, t) {
					for (var n, i, r = e.options,
					o = Q.makeArray(t), a = r.length; a--;) i = r[a],
					(i.selected = Q.inArray(i.value, o) >= 0) && (n = !0);
					return n || (e.selectedIndex = -1),
					o
				}
			}
		}
	}),
	Q.each(["radio", "checkbox"],
	function() {
		Q.valHooks[this] = {
			set: function(e, t) {
				return Q.isArray(t) ? e.checked = Q.inArray(Q(e).val(), t) >= 0 : void 0
			}
		},
		K.checkOn || (Q.valHooks[this].get = function(e) {
			return null === e.getAttribute("value") ? "on": e.value
		})
	}),
	Q.each("blur focus focusin focusout load resize scroll unload click dblclick mousedown mouseup mousemove mouseover mouseout mouseenter mouseleave change select submit keydown keypress keyup error contextmenu".split(" "),
	function(e, t) {
		Q.fn[t] = function(e, n) {
			return arguments.length > 0 ? this.on(t, null, e, n) : this.trigger(t)
		}
	}),
	Q.fn.extend({
		hover: function(e, t) {
			return this.mouseenter(e).mouseleave(t || e)
		},
		bind: function(e, t, n) {
			return this.on(e, null, t, n)
		},
		unbind: function(e, t) {
			return this.off(e, null, t)
		},
		delegate: function(e, t, n, i) {
			return this.on(t, e, n, i)
		},
		undelegate: function(e, t, n) {
			return 1 === arguments.length ? this.off(e, "**") : this.off(t, e || "**", n)
		}
	});
	var at = Q.now(),
	st = /\?/;
	Q.parseJSON = function(e) {
		return JSON.parse(e + "")
	},
	Q.parseXML = function(e) {
		var t, n;
		if (!e || "string" != typeof e) return null;
		try {
			n = new DOMParser,
			t = n.parseFromString(e, "text/xml")
		} catch(e) {
			t = void 0
		}
		return (!t || t.getElementsByTagName("parsererror").length) && Q.error("Invalid XML: " + e),
		t
	};
	var lt = /#.*$/,
	ct = /([?&])_=[^&]*/,
	ut = /^(.*?):[ \t]*([^\r\n]*)$/gm,
	dt = /^(?:about|app|app-storage|.+-extension|file|res|widget):$/,
	ht = /^(?:GET|HEAD)$/,
	ft = /^\/\//,
	pt = /^([\w.+-]+:)(?:\/\/(?:[^\/?#]*@|)([^\/?#:]*)(?::(\d+)|)|)/,
	mt = {},
	gt = {},
	vt = "*/".concat("*"),
	yt = e.location.href,
	bt = pt.exec(yt.toLowerCase()) || [];
	Q.extend({
		active: 0,
		lastModified: {},
		etag: {},
		ajaxSettings: {
			url: yt,
			type: "GET",
			isLocal: dt.test(bt[1]),
			global: !0,
			processData: !0,
			async: !0,
			contentType: "application/x-www-form-urlencoded; charset=UTF-8",
			accepts: {
				"*": vt,
				text: "text/plain",
				html: "text/html",
				xml: "application/xml, text/xml",
				json: "application/json, text/javascript"
			},
			contents: {
				xml: /xml/,
				html: /html/,
				json: /json/
			},
			responseFields: {
				xml: "responseXML",
				text: "responseText",
				json: "responseJSON"
			},
			converters: {
				"* text": String,
				"text html": !0,
				"text json": Q.parseJSON,
				"text xml": Q.parseXML
			},
			flatOptions: {
				url: !0,
				context: !0
			}
		},
		ajaxSetup: function(e, t) {
			return t ? j(j(e, Q.ajaxSettings), t) : j(Q.ajaxSettings, e)
		},
		ajaxPrefilter: $(mt),
		ajaxTransport: $(gt),
		ajax: function(e, t) {
			function n(e, t, n, a) {
				var l, u, v, y, w, x = t;
				2 !== b && (b = 2, s && clearTimeout(s), i = void 0, o = a || "", k.readyState = e > 0 ? 4 : 0, l = e >= 200 && 300 > e || 304 === e, n && (y = P(d, k, n)), y = I(d, y, k, l), l ? (d.ifModified && ((w = k.getResponseHeader("Last-Modified")) && (Q.lastModified[r] = w), (w = k.getResponseHeader("etag")) && (Q.etag[r] = w)), 204 === e || "HEAD" === d.type ? x = "nocontent": 304 === e ? x = "notmodified": (x = y.state, u = y.data, v = y.error, l = !v)) : (v = x, (e || !x) && (x = "error", 0 > e && (e = 0))), k.status = e, k.statusText = (t || x) + "", l ? p.resolveWith(h, [u, x, k]) : p.rejectWith(h, [k, x, v]), k.statusCode(g), g = void 0, c && f.trigger(l ? "ajaxSuccess": "ajaxError", [k, d, l ? u: v]), m.fireWith(h, [k, x]), c && (f.trigger("ajaxComplete", [k, d]), --Q.active || Q.event.trigger("ajaxStop")))
			}
			"object" == typeof e && (t = e, e = void 0),
			t = t || {};
			var i, r, o, a, s, l, c, u, d = Q.ajaxSetup({},
			t),
			h = d.context || d,
			f = d.context && (h.nodeType || h.jquery) ? Q(h) : Q.event,
			p = Q.Deferred(),
			m = Q.Callbacks("once memory"),
			g = d.statusCode || {},
			v = {},
			y = {},
			b = 0,
			w = "canceled",
			k = {
				readyState: 0,
				getResponseHeader: function(e) {
					var t;
					if (2 === b) {
						if (!a) for (a = {}; t = ut.exec(o);) a[t[1].toLowerCase()] = t[2];
						t = a[e.toLowerCase()]
					}
					return null == t ? null: t
				},
				getAllResponseHeaders: function() {
					return 2 === b ? o: null
				},
				setRequestHeader: function(e, t) {
					var n = e.toLowerCase();
					return b || (e = y[n] = y[n] || e, v[e] = t),
					this
				},
				overrideMimeType: function(e) {
					return b || (d.mimeType = e),
					this
				},
				statusCode: function(e) {
					var t;
					if (e) if (2 > b) for (t in e) g[t] = [g[t], e[t]];
					else k.always(e[k.status]);
					return this
				},
				abort: function(e) {
					var t = e || w;
					return i && i.abort(t),
					n(0, t),
					this
				}
			};
			if (p.promise(k).complete = m.add, k.success = k.done, k.error = k.fail, d.url = ((e || d.url || yt) + "").replace(lt, "").replace(ft, bt[1] + "//"), d.type = t.method || t.type || d.method || d.type, d.dataTypes = Q.trim(d.dataType || "*").toLowerCase().match(de) || [""], null == d.crossDomain && (l = pt.exec(d.url.toLowerCase()), d.crossDomain = !(!l || l[1] === bt[1] && l[2] === bt[2] && (l[3] || ("http:" === l[1] ? "80": "443")) === (bt[3] || ("http:" === bt[1] ? "80": "443")))), d.data && d.processData && "string" != typeof d.data && (d.data = Q.param(d.data, d.traditional)), q(mt, d, t, k), 2 === b) return k; (c = Q.event && d.global) && 0 == Q.active++&&Q.event.trigger("ajaxStart"),
			d.type = d.type.toUpperCase(),
			d.hasContent = !ht.test(d.type),
			r = d.url,
			d.hasContent || (d.data && (r = d.url += (st.test(r) ? "&": "?") + d.data, delete d.data), !1 === d.cache && (d.url = ct.test(r) ? r.replace(ct, "$1_=" + at++) : r + (st.test(r) ? "&": "?") + "_=" + at++)),
			d.ifModified && (Q.lastModified[r] && k.setRequestHeader("If-Modified-Since", Q.lastModified[r]), Q.etag[r] && k.setRequestHeader("If-None-Match", Q.etag[r])),
			(d.data && d.hasContent && !1 !== d.contentType || t.contentType) && k.setRequestHeader("Content-Type", d.contentType),
			k.setRequestHeader("Accept", d.dataTypes[0] && d.accepts[d.dataTypes[0]] ? d.accepts[d.dataTypes[0]] + ("*" !== d.dataTypes[0] ? ", " + vt + "; q=0.01": "") : d.accepts["*"]);
			for (u in d.headers) k.setRequestHeader(u, d.headers[u]);
			if (d.beforeSend && (!1 === d.beforeSend.call(h, k, d) || 2 === b)) return k.abort();
			w = "abort";
			for (u in {
				success: 1,
				error: 1,
				complete: 1
			}) k[u](d[u]);
			if (i = q(gt, d, t, k)) {
				k.readyState = 1,
				c && f.trigger("ajaxSend", [k, d]),
				d.async && d.timeout > 0 && (s = setTimeout(function() {
					k.abort("timeout")
				},
				d.timeout));
				try {
					b = 1,
					i.send(v, n)
				} catch(e) {
					if (! (2 > b)) throw e;
					n( - 1, e)
				}
			} else n( - 1, "No Transport");
			return k
		},
		getJSON: function(e, t, n) {
			return Q.get(e, t, n, "json")
		},
		getScript: function(e, t) {
			return Q.get(e, void 0, t, "script")
		}
	}),
	Q.each(["get", "post"],
	function(e, t) {
		Q[t] = function(e, n, i, r) {
			return Q.isFunction(n) && (r = r || i, i = n, n = void 0),
			Q.ajax({
				url: e,
				type: t,
				dataType: r,
				data: n,
				success: i
			})
		}
	}),
	Q._evalUrl = function(e) {
		return Q.ajax({
			url: e,
			type: "GET",
			dataType: "script",
			async: !1,
			global: !1,
			throws: !0
		})
	},
	Q.fn.extend({
		wrapAll: function(e) {
			var t;
			return Q.isFunction(e) ? this.each(function(t) {
				Q(this).wrapAll(e.call(this, t))
			}) : (this[0] && (t = Q(e, this[0].ownerDocument).eq(0).clone(!0), this[0].parentNode && t.insertBefore(this[0]), t.map(function() {
				for (var e = this; e.firstElementChild;) e = e.firstElementChild;
				return e
			}).append(this)), this)
		},
		wrapInner: function(e) {
			return this.each(Q.isFunction(e) ?
			function(t) {
				Q(this).wrapInner(e.call(this, t))
			}: function() {
				var t = Q(this),
				n = t.contents();
				n.length ? n.wrapAll(e) : t.append(e)
			})
		},
		wrap: function(e) {
			var t = Q.isFunction(e);
			return this.each(function(n) {
				Q(this).wrapAll(t ? e.call(this, n) : e)
			})
		},
		unwrap: function() {
			return this.parent().each(function() {
				Q.nodeName(this, "body") || Q(this).replaceWith(this.childNodes)
			}).end()
		}
	}),
	Q.expr.filters.hidden = function(e) {
		return e.offsetWidth <= 0 && e.offsetHeight <= 0
	},
	Q.expr.filters.visible = function(e) {
		return ! Q.expr.filters.hidden(e)
	};
	var wt = /%20/g,
	kt = /\[\]$/,
	xt = /\r?\n/g,
	_t = /^(?:submit|button|image|reset|file)$/i,
	Ct = /^(?:input|select|textarea|keygen)/i;
	Q.param = function(e, t) {
		var n, i = [],
		r = function(e, t) {
			t = Q.isFunction(t) ? t() : null == t ? "": t,
			i[i.length] = encodeURIComponent(e) + "=" + encodeURIComponent(t)
		};
		if (void 0 === t && (t = Q.ajaxSettings && Q.ajaxSettings.traditional), Q.isArray(e) || e.jquery && !Q.isPlainObject(e)) Q.each(e,
		function() {
			r(this.name, this.value)
		});
		else for (n in e) z(n, e[n], t, r);
		return i.join("&").replace(wt, "+")
	},
	Q.fn.extend({
		serialize: function() {
			return Q.param(this.serializeArray())
		},
		serializeArray: function() {
			return this.map(function() {
				var e = Q.prop(this, "elements");
				return e ? Q.makeArray(e) : this
			}).filter(function() {
				var e = this.type;
				return this.name && !Q(this).is(":disabled") && Ct.test(this.nodeName) && !_t.test(e) && (this.checked || !xe.test(e))
			}).map(function(e, t) {
				var n = Q(this).val();
				return null == n ? null: Q.isArray(n) ? Q.map(n,
				function(e) {
					return {
						name: t.name,
						value: e.replace(xt, "\r\n")
					}
				}) : {
					name: t.name,
					value: n.replace(xt, "\r\n")
				}
			}).get()
		}
	}),
	Q.ajaxSettings.xhr = function() {
		try {
			return new XMLHttpRequest
		} catch(e) {}
	};
	var St = 0,
	Mt = {},
	Tt = {
		0 : 200,
		1223 : 204
	},
	Dt = Q.ajaxSettings.xhr();
	e.attachEvent && e.attachEvent("onunload",
	function() {
		for (var e in Mt) Mt[e]()
	}),
	K.cors = !!Dt && "withCredentials" in Dt,
	K.ajax = Dt = !!Dt,
	Q.ajaxTransport(function(e) {
		var t;
		return K.cors || Dt && !e.crossDomain ? {
			send: function(n, i) {
				var r, o = e.xhr(),
				a = ++St;
				if (o.open(e.type, e.url, e.async, e.username, e.password), e.xhrFields) for (r in e.xhrFields) o[r] = e.xhrFields[r];
				e.mimeType && o.overrideMimeType && o.overrideMimeType(e.mimeType),
				e.crossDomain || n["X-Requested-With"] || (n["X-Requested-With"] = "XMLHttpRequest");
				for (r in n) o.setRequestHeader(r, n[r]);
				t = function(e) {
					return function() {
						t && (delete Mt[a], t = o.onload = o.onerror = null, "abort" === e ? o.abort() : "error" === e ? i(o.status, o.statusText) : i(Tt[o.status] || o.status, o.statusText, "string" == typeof o.responseText ? {
							text: o.responseText
						}: void 0, o.getAllResponseHeaders()))
					}
				},
				o.onload = t(),
				o.onerror = t("error"),
				t = Mt[a] = t("abort");
				try {
					o.send(e.hasContent && e.data || null)
				} catch(e) {
					if (t) throw e
				}
			},
			abort: function() {
				t && t()
			}
		}: void 0
	}),
	Q.ajaxSetup({
		accepts: {
			script: "text/javascript, application/javascript, application/ecmascript, application/x-ecmascript"
		},
		contents: {
			script: /(?:java|ecma)script/
		},
		converters: {
			"text script": function(e) {
				return Q.globalEval(e),
				e
			}
		}
	}),
	Q.ajaxPrefilter("script",
	function(e) {
		void 0 === e.cache && (e.cache = !1),
		e.crossDomain && (e.type = "GET")
	}),
	Q.ajaxTransport("script",
	function(e) {
		if (e.crossDomain) {
			var t, n;
			return {
				send: function(i, r) {
					t = Q("<script>").prop({
						async: !0,
						charset: e.scriptCharset,
						src: e.url
					}).on("load error", n = function(e) {
						t.remove(),
						n = null,
						e && r("error" === e.type ? 404 : 200, e.type)
					}),
					Z.head.appendChild(t[0])
				},
				abort: function() {
					n && n()
				}
			}
		}
	});
	var Lt = [],
	Ot = /(=)\?(?=&|$)|\?\?/;
	Q.ajaxSetup({
		jsonp: "callback",
		jsonpCallback: function() {
			var e = Lt.pop() || Q.expando + "_" + at++;
			return this[e] = !0,
			e
		}
	}),
	Q.ajaxPrefilter("json jsonp",
	function(t, n, i) {
		var r, o, a, s = !1 !== t.jsonp && (Ot.test(t.url) ? "url": "string" == typeof t.data && !(t.contentType || "").indexOf("application/x-www-form-urlencoded") && Ot.test(t.data) && "data");
		return s || "jsonp" === t.dataTypes[0] ? (r = t.jsonpCallback = Q.isFunction(t.jsonpCallback) ? t.jsonpCallback() : t.jsonpCallback, s ? t[s] = t[s].replace(Ot, "$1" + r) : !1 !== t.jsonp && (t.url += (st.test(t.url) ? "&": "?") + t.jsonp + "=" + r), t.converters["script json"] = function() {
			return a || Q.error(r + " was not called"),
			a[0]
		},
		t.dataTypes[0] = "json", o = e[r], e[r] = function() {
			a = arguments
		},
		i.always(function() {
			e[r] = o,
			t[r] && (t.jsonpCallback = n.jsonpCallback, Lt.push(r)),
			a && Q.isFunction(o) && o(a[0]),
			a = o = void 0
		}), "script") : void 0
	}),
	Q.parseHTML = function(e, t, n) {
		if (!e || "string" != typeof e) return null;
		"boolean" == typeof t && (n = t, t = !1),
		t = t || Z;
		var i = oe.exec(e),
		r = !n && [];
		return i ? [t.createElement(i[1])] : (i = Q.buildFragment([e], t, r), r && r.length && Q(r).remove(), Q.merge([], i.childNodes))
	};
	var Nt = Q.fn.load;
	Q.fn.load = function(e, t, n) {
		if ("string" != typeof e && Nt) return Nt.apply(this, arguments);
		var i, r, o, a = this,
		s = e.indexOf(" ");
		return s >= 0 && (i = Q.trim(e.slice(s)), e = e.slice(0, s)),
		Q.isFunction(t) ? (n = t, t = void 0) : t && "object" == typeof t && (r = "POST"),
		a.length > 0 && Q.ajax({
			url: e,
			type: r,
			dataType: "html",
			data: t
		}).done(function(e) {
			o = arguments,
			a.html(i ? Q("<div>").append(Q.parseHTML(e)).find(i) : e)
		}).complete(n &&
		function(e, t) {
			a.each(n, o || [e.responseText, t, e])
		}),
		this
	},
	Q.each(["ajaxStart", "ajaxStop", "ajaxComplete", "ajaxError", "ajaxSuccess", "ajaxSend"],
	function(e, t) {
		Q.fn[t] = function(e) {
			return this.on(t, e)
		}
	}),
	Q.expr.filters.animated = function(e) {
		return Q.grep(Q.timers,
		function(t) {
			return e === t.elem
		}).length
	};
	var At = e.document.documentElement;
	Q.offset = {
		setOffset: function(e, t, n) {
			var i, r, o, a, s, l, c = Q.css(e, "position"),
			u = Q(e),
			d = {};
			"static" === c && (e.style.position = "relative"),
			s = u.offset(),
			o = Q.css(e, "top"),
			l = Q.css(e, "left"),
			("absolute" === c || "fixed" === c) && (o + l).indexOf("auto") > -1 ? (i = u.position(), a = i.top, r = i.left) : (a = parseFloat(o) || 0, r = parseFloat(l) || 0),
			Q.isFunction(t) && (t = t.call(e, n, s)),
			null != t.top && (d.top = t.top - s.top + a),
			null != t.left && (d.left = t.left - s.left + r),
			"using" in t ? t.using.call(e, d) : u.css(d)
		}
	},
	Q.fn.extend({
		offset: function(e) {
			if (arguments.length) return void 0 === e ? this: this.each(function(t) {
				Q.offset.setOffset(this, e, t)
			});
			var t, n, i = this[0],
			r = {
				top: 0,
				left: 0
			},
			o = i && i.ownerDocument;
			return o ? (t = o.documentElement, Q.contains(t, i) ? (typeof i.getBoundingClientRect !== _e && (r = i.getBoundingClientRect()), n = W(o), {
				top: r.top + n.pageYOffset - t.clientTop,
				left: r.left + n.pageXOffset - t.clientLeft
			}) : r) : void 0
		},
		position: function() {
			if (this[0]) {
				var e, t, n = this[0],
				i = {
					top: 0,
					left: 0
				};
				return "fixed" === Q.css(n, "position") ? t = n.getBoundingClientRect() : (e = this.offsetParent(), t = this.offset(), Q.nodeName(e[0], "html") || (i = e.offset()), i.top += Q.css(e[0], "borderTopWidth", !0), i.left += Q.css(e[0], "borderLeftWidth", !0)),
				{
					top: t.top - i.top - Q.css(n, "marginTop", !0),
					left: t.left - i.left - Q.css(n, "marginLeft", !0)
				}
			}
		},
		offsetParent: function() {
			return this.map(function() {
				for (var e = this.offsetParent || At; e && !Q.nodeName(e, "html") && "static" === Q.css(e, "position");) e = e.offsetParent;
				return e || At
			})
		}
	}),
	Q.each({
		scrollLeft: "pageXOffset",
		scrollTop: "pageYOffset"
	},
	function(t, n) {
		var i = "pageYOffset" === n;
		Q.fn[t] = function(r) {
			return pe(this,
			function(t, r, o) {
				var a = W(t);
				return void 0 === o ? a ? a[n] : t[r] : void(a ? a.scrollTo(i ? e.pageXOffset: o, i ? o: e.pageYOffset) : t[r] = o)
			},
			t, r, arguments.length, null)
		}
	}),
	Q.each(["top", "left"],
	function(e, t) {
		Q.cssHooks[t] = x(K.pixelPosition,
		function(e, n) {
			return n ? (n = k(e, t), We.test(n) ? Q(e).position()[t] + "px": n) : void 0
		})
	}),
	Q.each({
		Height: "height",
		Width: "width"
	},
	function(e, t) {
		Q.each({
			padding: "inner" + e,
			content: t,
			"": "outer" + e
		},
		function(n, i) {
			Q.fn[i] = function(i, r) {
				var o = arguments.length && (n || "boolean" != typeof i),
				a = n || (!0 === i || !0 === r ? "margin": "border");
				return pe(this,
				function(t, n, i) {
					var r;
					return Q.isWindow(t) ? t.document.documentElement["client" + e] : 9 === t.nodeType ? (r = t.documentElement, Math.max(t.body["scroll" + e], r["scroll" + e], t.body["offset" + e], r["offset" + e], r["client" + e])) : void 0 === i ? Q.css(t, n, a) : Q.style(t, n, i, a)
				},
				t, o ? i: void 0, o, null)
			}
		})
	}),
	Q.fn.size = function() {
		return this.length
	},
	Q.fn.andSelf = Q.fn.addBack,
	"function" == typeof define && define.amd && define("jquery", [],
	function() {
		return Q
	});
	var Et = e.jQuery,
	$t = e.$;
	return Q.noConflict = function(t) {
		return e.$ === Q && (e.$ = $t),
		t && e.jQuery === Q && (e.jQuery = Et),
		Q
	},
	typeof t === _e && (e.jQuery = e.$ = Q),
	Q
}),
function() {
	function e(e) {
		function t(t, n, i, r, o, a) {
			for (; o >= 0 && a > o; o += e) {
				var s = r ? r[o] : o;
				i = n(i, t[s], s, t)
			}
			return i
		}
		return function(n, i, r, o) {
			i = b(i, o, 4);
			var a = !M(n) && y.keys(n),
			s = (a || n).length,
			l = e > 0 ? 0 : s - 1;
			return arguments.length < 3 && (r = n[a ? a[l] : l], l += e),
			t(n, i, r, a, l, s)
		}
	}
	function t(e) {
		return function(t, n, i) {
			n = w(n, i);
			for (var r = S(t), o = e > 0 ? 0 : r - 1; o >= 0 && r > o; o += e) if (n(t[o], o, t)) return o;
			return - 1
		}
	}
	function n(e, t, n) {
		return function(i, r, o) {
			var a = 0,
			s = S(i);
			if ("number" == typeof o) e > 0 ? a = o >= 0 ? o: Math.max(o + s, a) : s = o >= 0 ? Math.min(o + 1, s) : o + s + 1;
			else if (n && o && s) return o = n(i, r),
			i[o] === r ? o: -1;
			if (r !== r) return (o = t(u.call(i, a, s), y.isNaN)) >= 0 ? o + a: -1;
			for (o = e > 0 ? a: s - 1; o >= 0 && s > o; o += e) if (i[o] === r) return o;
			return - 1
		}
	}
	function i(e, t) {
		var n = N.length,
		i = e.constructor,
		r = y.isFunction(i) && i.prototype || s,
		o = "constructor";
		for (y.has(e, o) && !y.contains(t, o) && t.push(o); n--;)(o = N[n]) in e && e[o] !== r[o] && !y.contains(t, o) && t.push(o)
	}
	var r = this,
	o = r._,
	a = Array.prototype,
	s = Object.prototype,
	l = Function.prototype,
	c = a.push,
	u = a.slice,
	d = s.toString,
	h = s.hasOwnProperty,
	f = Array.isArray,
	p = Object.keys,
	m = l.bind,
	g = Object.create,
	v = function() {},
	y = function(e) {
		return e instanceof y ? e: this instanceof y ? void(this._wrapped = e) : new y(e)
	};
	"undefined" != typeof exports ? ("undefined" != typeof module && module.exports && (exports = module.exports = y), exports._ = y) : r._ = y,
	y.VERSION = "1.8.3";
	var b = function(e, t, n) {
		if (void 0 === t) return e;
		switch (null == n ? 3 : n) {
		case 1:
			return function(n) {
				return e.call(t, n)
			};
		case 2:
			return function(n, i) {
				return e.call(t, n, i)
			};
		case 3:
			return function(n, i, r) {
				return e.call(t, n, i, r)
			};
		case 4:
			return function(n, i, r, o) {
				return e.call(t, n, i, r, o)
			}
		}
		return function() {
			return e.apply(t, arguments)
		}
	},
	w = function(e, t, n) {
		return null == e ? y.identity: y.isFunction(e) ? b(e, t, n) : y.isObject(e) ? y.matcher(e) : y.property(e)
	};
	y.iteratee = function(e, t) {
		return w(e, t, 1 / 0)
	};
	var k = function(e, t) {
		return function(n) {
			var i = arguments.length;
			if (2 > i || null == n) return n;
			for (var r = 1; i > r; r++) for (var o = arguments[r], a = e(o), s = a.length, l = 0; s > l; l++) {
				var c = a[l];
				t && void 0 !== n[c] || (n[c] = o[c])
			}
			return n
		}
	},
	x = function(e) {
		if (!y.isObject(e)) return {};
		if (g) return g(e);
		v.prototype = e;
		var t = new v;
		return v.prototype = null,
		t
	},
	_ = function(e) {
		return function(t) {
			return null == t ? void 0 : t[e]
		}
	},
	C = Math.pow(2, 53) - 1,
	S = _("length"),
	M = function(e) {
		var t = S(e);
		return "number" == typeof t && t >= 0 && C >= t
	};
	y.each = y.forEach = function(e, t, n) {
		t = b(t, n);
		var i, r;
		if (M(e)) for (i = 0, r = e.length; r > i; i++) t(e[i], i, e);
		else {
			var o = y.keys(e);
			for (i = 0, r = o.length; r > i; i++) t(e[o[i]], o[i], e)
		}
		return e
	},
	y.map = y.collect = function(e, t, n) {
		t = w(t, n);
		for (var i = !M(e) && y.keys(e), r = (i || e).length, o = Array(r), a = 0; r > a; a++) {
			var s = i ? i[a] : a;
			o[a] = t(e[s], s, e)
		}
		return o
	},
	y.reduce = y.foldl = y.inject = e(1),
	y.reduceRight = y.foldr = e( - 1),
	y.find = y.detect = function(e, t, n) {
		var i;
		return void 0 !== (i = M(e) ? y.findIndex(e, t, n) : y.findKey(e, t, n)) && -1 !== i ? e[i] : void 0
	},
	y.filter = y.select = function(e, t, n) {
		var i = [];
		return t = w(t, n),
		y.each(e,
		function(e, n, r) {
			t(e, n, r) && i.push(e)
		}),
		i
	},
	y.reject = function(e, t, n) {
		return y.filter(e, y.negate(w(t)), n)
	},
	y.every = y.all = function(e, t, n) {
		t = w(t, n);
		for (var i = !M(e) && y.keys(e), r = (i || e).length, o = 0; r > o; o++) {
			var a = i ? i[o] : o;
			if (!t(e[a], a, e)) return ! 1
		}
		return ! 0
	},
	y.some = y.any = function(e, t, n) {
		t = w(t, n);
		for (var i = !M(e) && y.keys(e), r = (i || e).length, o = 0; r > o; o++) {
			var a = i ? i[o] : o;
			if (t(e[a], a, e)) return ! 0
		}
		return ! 1
	},
	y.contains = y.includes = y.include = function(e, t, n, i) {
		return M(e) || (e = y.values(e)),
		("number" != typeof n || i) && (n = 0),
		y.indexOf(e, t, n) >= 0
	},
	y.invoke = function(e, t) {
		var n = u.call(arguments, 2),
		i = y.isFunction(t);
		return y.map(e,
		function(e) {
			var r = i ? t: e[t];
			return null == r ? r: r.apply(e, n)
		})
	},
	y.pluck = function(e, t) {
		return y.map(e, y.property(t))
	},
	y.where = function(e, t) {
		return y.filter(e, y.matcher(t))
	},
	y.findWhere = function(e, t) {
		return y.find(e, y.matcher(t))
	},
	y.max = function(e, t, n) {
		var i, r, o = -1 / 0,
		a = -1 / 0;
		if (null == t && null != e) for (var s = 0,
		l = (e = M(e) ? e: y.values(e)).length; l > s; s++)(i = e[s]) > o && (o = i);
		else t = w(t, n),
		y.each(e,
		function(e, n, i) { ((r = t(e, n, i)) > a || r === -1 / 0 && o === -1 / 0) && (o = e, a = r)
		});
		return o
	},
	y.min = function(e, t, n) {
		var i, r, o = 1 / 0,
		a = 1 / 0;
		if (null == t && null != e) for (var s = 0,
		l = (e = M(e) ? e: y.values(e)).length; l > s; s++) i = e[s],
		o > i && (o = i);
		else t = w(t, n),
		y.each(e,
		function(e, n, i) {
			r = t(e, n, i),
			(a > r || 1 / 0 === r && 1 / 0 === o) && (o = e, a = r)
		});
		return o
	},
	y.shuffle = function(e) {
		for (var t, n = M(e) ? e: y.values(e), i = n.length, r = Array(i), o = 0; i > o; o++)(t = y.random(0, o)) !== o && (r[o] = r[t]),
		r[t] = n[o];
		return r
	},
	y.sample = function(e, t, n) {
		return null == t || n ? (M(e) || (e = y.values(e)), e[y.random(e.length - 1)]) : y.shuffle(e).slice(0, Math.max(0, t))
	},
	y.sortBy = function(e, t, n) {
		return t = w(t, n),
		y.pluck(y.map(e,
		function(e, n, i) {
			return {
				value: e,
				index: n,
				criteria: t(e, n, i)
			}
		}).sort(function(e, t) {
			var n = e.criteria,
			i = t.criteria;
			if (n !== i) {
				if (n > i || void 0 === n) return 1;
				if (i > n || void 0 === i) return - 1
			}
			return e.index - t.index
		}), "value")
	};
	var T = function(e) {
		return function(t, n, i) {
			var r = {};
			return n = w(n, i),
			y.each(t,
			function(i, o) {
				var a = n(i, o, t);
				e(r, i, a)
			}),
			r
		}
	};
	y.groupBy = T(function(e, t, n) {
		y.has(e, n) ? e[n].push(t) : e[n] = [t]
	}),
	y.indexBy = T(function(e, t, n) {
		e[n] = t
	}),
	y.countBy = T(function(e, t, n) {
		y.has(e, n) ? e[n]++:e[n] = 1
	}),
	y.toArray = function(e) {
		return e ? y.isArray(e) ? u.call(e) : M(e) ? y.map(e, y.identity) : y.values(e) : []
	},
	y.size = function(e) {
		return null == e ? 0 : M(e) ? e.length: y.keys(e).length
	},
	y.partition = function(e, t, n) {
		t = w(t, n);
		var i = [],
		r = [];
		return y.each(e,
		function(e, n, o) { (t(e, n, o) ? i: r).push(e)
		}),
		[i, r]
	},
	y.first = y.head = y.take = function(e, t, n) {
		return null == e ? void 0 : null == t || n ? e[0] : y.initial(e, e.length - t)
	},
	y.initial = function(e, t, n) {
		return u.call(e, 0, Math.max(0, e.length - (null == t || n ? 1 : t)))
	},
	y.last = function(e, t, n) {
		return null == e ? void 0 : null == t || n ? e[e.length - 1] : y.rest(e, Math.max(0, e.length - t))
	},
	y.rest = y.tail = y.drop = function(e, t, n) {
		return u.call(e, null == t || n ? 1 : t)
	},
	y.compact = function(e) {
		return y.filter(e, y.identity)
	};
	var D = function(e, t, n, i) {
		for (var r = [], o = 0, a = i || 0, s = S(e); s > a; a++) {
			var l = e[a];
			if (M(l) && (y.isArray(l) || y.isArguments(l))) {
				t || (l = D(l, t, n));
				var c = 0,
				u = l.length;
				for (r.length += u; u > c;) r[o++] = l[c++]
			} else n || (r[o++] = l)
		}
		return r
	};
	y.flatten = function(e, t) {
		return D(e, t, !1)
	},
	y.without = function(e) {
		return y.difference(e, u.call(arguments, 1))
	},
	y.uniq = y.unique = function(e, t, n, i) {
		y.isBoolean(t) || (i = n, n = t, t = !1),
		null != n && (n = w(n, i));
		for (var r = [], o = [], a = 0, s = S(e); s > a; a++) {
			var l = e[a],
			c = n ? n(l, a, e) : l;
			t ? (a && o === c || r.push(l), o = c) : n ? y.contains(o, c) || (o.push(c), r.push(l)) : y.contains(r, l) || r.push(l)
		}
		return r
	},
	y.union = function() {
		return y.uniq(D(arguments, !0, !0))
	},
	y.intersection = function(e) {
		for (var t = [], n = arguments.length, i = 0, r = S(e); r > i; i++) {
			var o = e[i];
			if (!y.contains(t, o)) {
				for (var a = 1; n > a && y.contains(arguments[a], o); a++);
				a === n && t.push(o)
			}
		}
		return t
	},
	y.difference = function(e) {
		var t = D(arguments, !0, !0, 1);
		return y.filter(e,
		function(e) {
			return ! y.contains(t, e)
		})
	},
	y.zip = function() {
		return y.unzip(arguments)
	},
	y.unzip = function(e) {
		for (var t = e && y.max(e, S).length || 0, n = Array(t), i = 0; t > i; i++) n[i] = y.pluck(e, i);
		return n
	},
	y.object = function(e, t) {
		for (var n = {},
		i = 0,
		r = S(e); r > i; i++) t ? n[e[i]] = t[i] : n[e[i][0]] = e[i][1];
		return n
	},
	y.findIndex = t(1),
	y.findLastIndex = t( - 1),
	y.sortedIndex = function(e, t, n, i) {
		for (var r = (n = w(n, i, 1))(t), o = 0, a = S(e); a > o;) {
			var s = Math.floor((o + a) / 2);
			n(e[s]) < r ? o = s + 1 : a = s
		}
		return o
	},
	y.indexOf = n(1, y.findIndex, y.sortedIndex),
	y.lastIndexOf = n( - 1, y.findLastIndex),
	y.range = function(e, t, n) {
		null == t && (t = e || 0, e = 0),
		n = n || 1;
		for (var i = Math.max(Math.ceil((t - e) / n), 0), r = Array(i), o = 0; i > o; o++, e += n) r[o] = e;
		return r
	};
	var L = function(e, t, n, i, r) {
		if (! (i instanceof t)) return e.apply(n, r);
		var o = x(e.prototype),
		a = e.apply(o, r);
		return y.isObject(a) ? a: o
	};
	y.bind = function(e, t) {
		if (m && e.bind === m) return m.apply(e, u.call(arguments, 1));
		if (!y.isFunction(e)) throw new TypeError("Bind must be called on a function");
		var n = u.call(arguments, 2),
		i = function() {
			return L(e, i, t, this, n.concat(u.call(arguments)))
		};
		return i
	},
	y.partial = function(e) {
		var t = u.call(arguments, 1),
		n = function() {
			for (var i = 0,
			r = t.length,
			o = Array(r), a = 0; r > a; a++) o[a] = t[a] === y ? arguments[i++] : t[a];
			for (; i < arguments.length;) o.push(arguments[i++]);
			return L(e, n, this, this, o)
		};
		return n
	},
	y.bindAll = function(e) {
		var t, n, i = arguments.length;
		if (1 >= i) throw new Error("bindAll must be passed function names");
		for (t = 1; i > t; t++) n = arguments[t],
		e[n] = y.bind(e[n], e);
		return e
	},
	y.memoize = function(e, t) {
		var n = function(i) {
			var r = n.cache,
			o = "" + (t ? t.apply(this, arguments) : i);
			return y.has(r, o) || (r[o] = e.apply(this, arguments)),
			r[o]
		};
		return n.cache = {},
		n
	},
	y.delay = function(e, t) {
		var n = u.call(arguments, 2);
		return setTimeout(function() {
			return e.apply(null, n)
		},
		t)
	},
	y.defer = y.partial(y.delay, y, 1),
	y.throttle = function(e, t, n) {
		var i, r, o, a = null,
		s = 0;
		n || (n = {});
		var l = function() {
			s = !1 === n.leading ? 0 : y.now(),
			a = null,
			o = e.apply(i, r),
			a || (i = r = null)
		};
		return function() {
			var c = y.now();
			s || !1 !== n.leading || (s = c);
			var u = t - (c - s);
			return i = this,
			r = arguments,
			0 >= u || u > t ? (a && (clearTimeout(a), a = null), s = c, o = e.apply(i, r), a || (i = r = null)) : a || !1 === n.trailing || (a = setTimeout(l, u)),
			o
		}
	},
	y.debounce = function(e, t, n) {
		var i, r, o, a, s, l = function() {
			var c = y.now() - a;
			t > c && c >= 0 ? i = setTimeout(l, t - c) : (i = null, n || (s = e.apply(o, r), i || (o = r = null)))
		};
		return function() {
			o = this,
			r = arguments,
			a = y.now();
			var c = n && !i;
			return i || (i = setTimeout(l, t)),
			c && (s = e.apply(o, r), o = r = null),
			s
		}
	},
	y.wrap = function(e, t) {
		return y.partial(t, e)
	},
	y.negate = function(e) {
		return function() {
			return ! e.apply(this, arguments)
		}
	},
	y.compose = function() {
		var e = arguments,
		t = e.length - 1;
		return function() {
			for (var n = t,
			i = e[t].apply(this, arguments); n--;) i = e[n].call(this, i);
			return i
		}
	},
	y.after = function(e, t) {
		return function() {
			return--e < 1 ? t.apply(this, arguments) : void 0
		}
	},
	y.before = function(e, t) {
		var n;
		return function() {
			return--e > 0 && (n = t.apply(this, arguments)),
			1 >= e && (t = null),
			n
		}
	},
	y.once = y.partial(y.before, 2);
	var O = !{
		toString: null
	}.propertyIsEnumerable("toString"),
	N = ["valueOf", "isPrototypeOf", "toString", "propertyIsEnumerable", "hasOwnProperty", "toLocaleString"];
	y.keys = function(e) {
		if (!y.isObject(e)) return [];
		if (p) return p(e);
		var t = [];
		for (var n in e) y.has(e, n) && t.push(n);
		return O && i(e, t),
		t
	},
	y.allKeys = function(e) {
		if (!y.isObject(e)) return [];
		var t = [];
		for (var n in e) t.push(n);
		return O && i(e, t),
		t
	},
	y.values = function(e) {
		for (var t = y.keys(e), n = t.length, i = Array(n), r = 0; n > r; r++) i[r] = e[t[r]];
		return i
	},
	y.mapObject = function(e, t, n) {
		t = w(t, n);
		for (var i, r = y.keys(e), o = r.length, a = {},
		s = 0; o > s; s++) i = r[s],
		a[i] = t(e[i], i, e);
		return a
	},
	y.pairs = function(e) {
		for (var t = y.keys(e), n = t.length, i = Array(n), r = 0; n > r; r++) i[r] = [t[r], e[t[r]]];
		return i
	},
	y.invert = function(e) {
		for (var t = {},
		n = y.keys(e), i = 0, r = n.length; r > i; i++) t[e[n[i]]] = n[i];
		return t
	},
	y.functions = y.methods = function(e) {
		var t = [];
		for (var n in e) y.isFunction(e[n]) && t.push(n);
		return t.sort()
	},
	y.extend = k(y.allKeys),
	y.extendOwn = y.assign = k(y.keys),
	y.findKey = function(e, t, n) {
		t = w(t, n);
		for (var i, r = y.keys(e), o = 0, a = r.length; a > o; o++) if (i = r[o], t(e[i], i, e)) return i
	},
	y.pick = function(e, t, n) {
		var i, r, o = {},
		a = e;
		if (null == a) return o;
		y.isFunction(t) ? (r = y.allKeys(a), i = b(t, n)) : (r = D(arguments, !1, !1, 1), i = function(e, t, n) {
			return t in n
		},
		a = Object(a));
		for (var s = 0,
		l = r.length; l > s; s++) {
			var c = r[s],
			u = a[c];
			i(u, c, a) && (o[c] = u)
		}
		return o
	},
	y.omit = function(e, t, n) {
		if (y.isFunction(t)) t = y.negate(t);
		else {
			var i = y.map(D(arguments, !1, !1, 1), String);
			t = function(e, t) {
				return ! y.contains(i, t)
			}
		}
		return y.pick(e, t, n)
	},
	y.defaults = k(y.allKeys, !0),
	y.create = function(e, t) {
		var n = x(e);
		return t && y.extendOwn(n, t),
		n
	},
	y.clone = function(e) {
		return y.isObject(e) ? y.isArray(e) ? e.slice() : y.extend({},
		e) : e
	},
	y.tap = function(e, t) {
		return t(e),
		e
	},
	y.isMatch = function(e, t) {
		var n = y.keys(t),
		i = n.length;
		if (null == e) return ! i;
		for (var r = Object(e), o = 0; i > o; o++) {
			var a = n[o];
			if (t[a] !== r[a] || !(a in r)) return ! 1
		}
		return ! 0
	};
	var A = function(e, t, n, i) {
		if (e === t) return 0 !== e || 1 / e == 1 / t;
		if (null == e || null == t) return e === t;
		e instanceof y && (e = e._wrapped),
		t instanceof y && (t = t._wrapped);
		var r = d.call(e);
		if (r !== d.call(t)) return ! 1;
		switch (r) {
		case "[object RegExp]":
		case "[object String]":
			return "" + e == "" + t;
		case "[object Number]":
			return + e != +e ? +t != +t: 0 == +e ? 1 / +e == 1 / t: +e == +t;
		case "[object Date]":
		case "[object Boolean]":
			return + e == +t
		}
		var o = "[object Array]" === r;
		if (!o) {
			if ("object" != typeof e || "object" != typeof t) return ! 1;
			var a = e.constructor,
			s = t.constructor;
			if (a !== s && !(y.isFunction(a) && a instanceof a && y.isFunction(s) && s instanceof s) && "constructor" in e && "constructor" in t) return ! 1
		}
		n = n || [],
		i = i || [];
		for (var l = n.length; l--;) if (n[l] === e) return i[l] === t;
		if (n.push(e), i.push(t), o) {
			if ((l = e.length) !== t.length) return ! 1;
			for (; l--;) if (!A(e[l], t[l], n, i)) return ! 1
		} else {
			var c, u = y.keys(e);
			if (l = u.length, y.keys(t).length !== l) return ! 1;
			for (; l--;) if (c = u[l], !y.has(t, c) || !A(e[c], t[c], n, i)) return ! 1
		}
		return n.pop(),
		i.pop(),
		!0
	};
	y.isEqual = function(e, t) {
		return A(e, t)
	},
	y.isEmpty = function(e) {
		return null == e || (M(e) && (y.isArray(e) || y.isString(e) || y.isArguments(e)) ? 0 === e.length: 0 === y.keys(e).length)
	},
	y.isElement = function(e) {
		return ! (!e || 1 !== e.nodeType)
	},
	y.isArray = f ||
	function(e) {
		return "[object Array]" === d.call(e)
	},
	y.isObject = function(e) {
		var t = typeof e;
		return "function" === t || "object" === t && !!e
	},
	y.each(["Arguments", "Function", "String", "Number", "Date", "RegExp", "Error"],
	function(e) {
		y["is" + e] = function(t) {
			return d.call(t) === "[object " + e + "]"
		}
	}),
	y.isArguments(arguments) || (y.isArguments = function(e) {
		return y.has(e, "callee")
	}),
	"function" != typeof / . / &&"object" != typeof Int8Array && (y.isFunction = function(e) {
		return "function" == typeof e || !1
	}),
	y.isFinite = function(e) {
		return isFinite(e) && !isNaN(parseFloat(e))
	},
	y.isNaN = function(e) {
		return y.isNumber(e) && e !== +e
	},
	y.isBoolean = function(e) {
		return ! 0 === e || !1 === e || "[object Boolean]" === d.call(e)
	},
	y.isNull = function(e) {
		return null === e
	},
	y.isUndefined = function(e) {
		return void 0 === e
	},
	y.has = function(e, t) {
		return null != e && h.call(e, t)
	},
	y.noConflict = function() {
		return r._ = o,
		this
	},
	y.identity = function(e) {
		return e
	},
	y.constant = function(e) {
		return function() {
			return e
		}
	},
	y.noop = function() {},
	y.property = _,
	y.propertyOf = function(e) {
		return null == e ?
		function() {}: function(t) {
			return e[t]
		}
	},
	y.matcher = y.matches = function(e) {
		return e = y.extendOwn({},
		e),
		function(t) {
			return y.isMatch(t, e)
		}
	},
	y.times = function(e, t, n) {
		var i = Array(Math.max(0, e));
		t = b(t, n, 1);
		for (var r = 0; e > r; r++) i[r] = t(r);
		return i
	},
	y.random = function(e, t) {
		return null == t && (t = e, e = 0),
		e + Math.floor(Math.random() * (t - e + 1))
	},
	y.now = Date.now ||
	function() {
		return (new Date).getTime()
	};
	var E = {
		"&": "&amp;",
		"<": "&lt;",
		">": "&gt;",
		'"': "&quot;",
		"'": "&#x27;",
		"`": "&#x60;"
	},
	$ = y.invert(E),
	q = function(e) {
		var t = function(t) {
			return e[t]
		},
		n = "(?:" + y.keys(e).join("|") + ")",
		i = RegExp(n),
		r = RegExp(n, "g");
		return function(e) {
			return e = null == e ? "": "" + e,
			i.test(e) ? e.replace(r, t) : e
		}
	};
	y.escape = q(E),
	y.unescape = q($),
	y.result = function(e, t, n) {
		var i = null == e ? void 0 : e[t];
		return void 0 === i && (i = n),
		y.isFunction(i) ? i.call(e) : i
	};
	var j = 0;
	y.uniqueId = function(e) {
		var t = ++j + "";
		return e ? e + t: t
	},
	y.templateSettings = {
		evaluate: /<%([\s\S]+?)%>/g,
		interpolate: /<%=([\s\S]+?)%>/g,
		escape: /<%-([\s\S]+?)%>/g
	};
	var P = /(.)^/,
	I = {
		"'": "'",
		"\\": "\\",
		"\r": "r",
		"\n": "n",
		"\u2028": "u2028",
		"\u2029": "u2029"
	},
	z = /\\|'|\r|\n|\u2028|\u2029/g,
	W = function(e) {
		return "\\" + I[e]
	};
	y.template = function(e, t, n) { ! t && n && (t = n),
		t = y.defaults({},
		t, y.templateSettings);
		var i = RegExp([(t.escape || P).source, (t.interpolate || P).source, (t.evaluate || P).source].join("|") + "|$", "g"),
		r = 0,
		o = "__p+='";
		e.replace(i,
		function(t, n, i, a, s) {
			return o += e.slice(r, s).replace(z, W),
			r = s + t.length,
			n ? o += "'+\n((__t=(" + n + "))==null?'':_.escape(__t))+\n'": i ? o += "'+\n((__t=(" + i + "))==null?'':__t)+\n'": a && (o += "';\n" + a + "\n__p+='"),
			t
		}),
		o += "';\n",
		t.variable || (o = "with(obj||{}){\n" + o + "}\n"),
		o = "var __t,__p='',__j=Array.prototype.join,print=function(){__p+=__j.call(arguments,'');};\n" + o + "return __p;\n";
		try {
			var a = new Function(t.variable || "obj", "_", o)
		} catch(e) {
			throw e.source = o,
			e
		}
		var s = function(e) {
			return a.call(this, e, y)
		},
		l = t.variable || "obj";
		return s.source = "function(" + l + "){\n" + o + "}",
		s
	},
	y.chain = function(e) {
		var t = y(e);
		return t._chain = !0,
		t
	};
	var H = function(e, t) {
		return e._chain ? y(t).chain() : t
	};
	y.mixin = function(e) {
		y.each(y.functions(e),
		function(t) {
			var n = y[t] = e[t];
			y.prototype[t] = function() {
				var e = [this._wrapped];
				return c.apply(e, arguments),
				H(this, n.apply(y, e))
			}
		})
	},
	y.mixin(y),
	y.each(["pop", "push", "reverse", "shift", "sort", "splice", "unshift"],
	function(e) {
		var t = a[e];
		y.prototype[e] = function() {
			var n = this._wrapped;
			return t.apply(n, arguments),
			"shift" !== e && "splice" !== e || 0 !== n.length || delete n[0],
			H(this, n)
		}
	}),
	y.each(["concat", "join", "slice"],
	function(e) {
		var t = a[e];
		y.prototype[e] = function() {
			return H(this, t.apply(this._wrapped, arguments))
		}
	}),
	y.prototype.value = function() {
		return this._wrapped
	},
	y.prototype.valueOf = y.prototype.toJSON = y.prototype.value,
	y.prototype.toString = function() {
		return "" + this._wrapped
	},
	"function" == typeof define && define.amd && define("underscore", [],
	function() {
		return y
	})
}.call(this),
function(e, t) {
	"object" == typeof exports && "undefined" != typeof module ? module.exports = t() : "function" == typeof define && define.amd ? define(t) : e.moment = t()
} (this,
function() {
	"use strict";
	function e() {
		return yt.apply(null, arguments)
	}
	function t(e) {
		return e instanceof Array || "[object Array]" === Object.prototype.toString.call(e)
	}
	function n(e) {
		return null != e && "[object Object]" === Object.prototype.toString.call(e)
	}
	function i(e) {
		var t;
		for (t in e) return ! 1;
		return ! 0
	}
	function r(e) {
		return void 0 === e
	}
	function o(e) {
		return "number" == typeof e || "[object Number]" === Object.prototype.toString.call(e)
	}
	function a(e) {
		return e instanceof Date || "[object Date]" === Object.prototype.toString.call(e)
	}
	function s(e, t) {
		var n, i = [];
		for (n = 0; n < e.length; ++n) i.push(t(e[n], n));
		return i
	}
	function l(e, t) {
		return Object.prototype.hasOwnProperty.call(e, t)
	}
	function c(e, t) {
		for (var n in t) l(t, n) && (e[n] = t[n]);
		return l(t, "toString") && (e.toString = t.toString),
		l(t, "valueOf") && (e.valueOf = t.valueOf),
		e
	}
	function u(e, t, n, i) {
		return qe(e, t, n, i, !0).utc()
	}
	function d() {
		return {
			empty: !1,
			unusedTokens: [],
			unusedInput: [],
			overflow: -2,
			charsLeftOver: 0,
			nullInput: !1,
			invalidMonth: null,
			invalidFormat: !1,
			userInvalidated: !1,
			iso: !1,
			parsedDateParts: [],
			meridiem: null,
			rfc2822: !1,
			weekdayMismatch: !1
		}
	}
	function h(e) {
		return null == e._pf && (e._pf = d()),
		e._pf
	}
	function f(e) {
		if (null == e._isValid) {
			var t = h(e),
			n = wt.call(t.parsedDateParts,
			function(e) {
				return null != e
			}),
			i = !isNaN(e._d.getTime()) && t.overflow < 0 && !t.empty && !t.invalidMonth && !t.invalidWeekday && !t.nullInput && !t.invalidFormat && !t.userInvalidated && (!t.meridiem || t.meridiem && n);
			if (e._strict && (i = i && 0 === t.charsLeftOver && 0 === t.unusedTokens.length && void 0 === t.bigHour), null != Object.isFrozen && Object.isFrozen(e)) return i;
			e._isValid = i
		}
		return e._isValid
	}
	function p(e) {
		var t = u(NaN);
		return null != e ? c(h(t), e) : h(t).userInvalidated = !0,
		t
	}
	function m(e, t) {
		var n, i, o;
		if (r(t._isAMomentObject) || (e._isAMomentObject = t._isAMomentObject), r(t._i) || (e._i = t._i), r(t._f) || (e._f = t._f), r(t._l) || (e._l = t._l), r(t._strict) || (e._strict = t._strict), r(t._tzm) || (e._tzm = t._tzm), r(t._isUTC) || (e._isUTC = t._isUTC), r(t._offset) || (e._offset = t._offset), r(t._pf) || (e._pf = h(t)), r(t._locale) || (e._locale = t._locale), kt.length > 0) for (n = 0; n < kt.length; n++) i = kt[n],
		o = t[i],
		r(o) || (e[i] = o);
		return e
	}
	function g(t) {
		m(this, t),
		this._d = new Date(null != t._d ? t._d.getTime() : NaN),
		this.isValid() || (this._d = new Date(NaN)),
		!1 === xt && (xt = !0, e.updateOffset(this), xt = !1)
	}
	function v(e) {
		return e instanceof g || null != e && null != e._isAMomentObject
	}
	function y(e) {
		return e < 0 ? Math.ceil(e) || 0 : Math.floor(e)
	}
	function b(e) {
		var t = +e,
		n = 0;
		return 0 !== t && isFinite(t) && (n = y(t)),
		n
	}
	function w(e, t, n) {
		var i, r = Math.min(e.length, t.length),
		o = Math.abs(e.length - t.length),
		a = 0;
		for (i = 0; i < r; i++)(n && e[i] !== t[i] || !n && b(e[i]) !== b(t[i])) && a++;
		return a + o
	}
	function k(t) { ! 1 === e.suppressDeprecationWarnings && "undefined" != typeof console && console.warn && console.warn("Deprecation warning: " + t)
	}
	function x(t, n) {
		var i = !0;
		return c(function() {
			if (null != e.deprecationHandler && e.deprecationHandler(null, t), i) {
				for (var r, o = [], a = 0; a < arguments.length; a++) {
					if (r = "", "object" == typeof arguments[a]) {
						r += "\n[" + a + "] ";
						for (var s in arguments[0]) r += s + ": " + arguments[0][s] + ", ";
						r = r.slice(0, -2)
					} else r = arguments[a];
					o.push(r)
				}
				k(t + "\nArguments: " + Array.prototype.slice.call(o).join("") + "\n" + (new Error).stack),
				i = !1
			}
			return n.apply(this, arguments)
		},
		n)
	}
	function _(t, n) {
		null != e.deprecationHandler && e.deprecationHandler(t, n),
		_t[t] || (k(n), _t[t] = !0)
	}
	function C(e) {
		return e instanceof Function || "[object Function]" === Object.prototype.toString.call(e)
	}
	function S(e, t) {
		var i, r = c({},
		e);
		for (i in t) l(t, i) && (n(e[i]) && n(t[i]) ? (r[i] = {},
		c(r[i], e[i]), c(r[i], t[i])) : null != t[i] ? r[i] = t[i] : delete r[i]);
		for (i in e) l(e, i) && !l(t, i) && n(e[i]) && (r[i] = c({},
		r[i]));
		return r
	}
	function M(e) {
		null != e && this.set(e)
	}
	function T(e, t) {
		var n = e.toLowerCase();
		Nt[n] = Nt[n + "s"] = Nt[t] = e
	}
	function D(e) {
		return "string" == typeof e ? Nt[e] || Nt[e.toLowerCase()] : void 0
	}
	function L(e) {
		var t, n, i = {};
		for (n in e) l(e, n) && (t = D(n)) && (i[t] = e[n]);
		return i
	}
	function O(e, t) {
		At[e] = t
	}
	function N(e) {
		var t = [];
		for (var n in e) t.push({
			unit: n,
			priority: At[n]
		});
		return t.sort(function(e, t) {
			return e.priority - t.priority
		}),
		t
	}
	function A(t, n) {
		return function(i) {
			return null != i ? ($(this, t, i), e.updateOffset(this, n), this) : E(this, t)
		}
	}
	function E(e, t) {
		return e.isValid() ? e._d["get" + (e._isUTC ? "UTC": "") + t]() : NaN
	}
	function $(e, t, n) {
		e.isValid() && e._d["set" + (e._isUTC ? "UTC": "") + t](n)
	}
	function q(e, t, n) {
		var i = "" + Math.abs(e),
		r = t - i.length;
		return (e >= 0 ? n ? "+": "": "-") + Math.pow(10, Math.max(0, r)).toString().substr(1) + i
	}
	function j(e, t, n, i) {
		var r = i;
		"string" == typeof i && (r = function() {
			return this[i]()
		}),
		e && (jt[e] = r),
		t && (jt[t[0]] = function() {
			return q(r.apply(this, arguments), t[1], t[2])
		}),
		n && (jt[n] = function() {
			return this.localeData().ordinal(r.apply(this, arguments), e)
		})
	}
	function P(e) {
		return e.match(/\[[\s\S]/) ? e.replace(/^\[|\]$/g, "") : e.replace(/\\/g, "")
	}
	function I(e) {
		var t, n, i = e.match(Et);
		for (t = 0, n = i.length; t < n; t++) jt[i[t]] ? i[t] = jt[i[t]] : i[t] = P(i[t]);
		return function(t) {
			var r, o = "";
			for (r = 0; r < n; r++) o += C(i[r]) ? i[r].call(t, e) : i[r];
			return o
		}
	}
	function z(e, t) {
		return e.isValid() ? (t = W(t, e.localeData()), qt[t] = qt[t] || I(t), qt[t](e)) : e.localeData().invalidDate()
	}
	function W(e, t) {
		var n = 5;
		for ($t.lastIndex = 0; n >= 0 && $t.test(e);) e = e.replace($t,
		function(e) {
			return t.longDateFormat(e) || e
		}),
		$t.lastIndex = 0,
		n -= 1;
		return e
	}
	function H(e, t, n) {
		en[e] = C(t) ? t: function(e, i) {
			return e && n ? n: t
		}
	}
	function F(e, t) {
		return l(en, e) ? en[e](t._strict, t._locale) : new RegExp(Y(e))
	}
	function Y(e) {
		return R(e.replace("\\", "").replace(/\\(\[)|\\(\])|\[([^\]\[]*)\]|\\(.)/g,
		function(e, t, n, i, r) {
			return t || n || i || r
		}))
	}
	function R(e) {
		return e.replace(/[-\/\\^$*+?.()|[\]{}]/g, "\\$&")
	}
	function U(e, t) {
		var n, i = t;
		for ("string" == typeof e && (e = [e]), o(t) && (i = function(e, n) {
			n[t] = b(e)
		}), n = 0; n < e.length; n++) tn[e[n]] = i
	}
	function B(e, t) {
		U(e,
		function(e, n, i, r) {
			i._w = i._w || {},
			t(e, i._w, i, r)
		})
	}
	function G(e, t, n) {
		null != t && l(tn, e) && tn[e](t, n._a, n, e)
	}
	function V(e, t) {
		return new Date(Date.UTC(e, t + 1, 0)).getUTCDate()
	}
	function K(e, t, n) {
		var i, r, o, a = e.toLocaleLowerCase();
		if (!this._monthsParse) for (this._monthsParse = [], this._longMonthsParse = [], this._shortMonthsParse = [], i = 0; i < 12; ++i) o = u([2e3, i]),
		this._shortMonthsParse[i] = this.monthsShort(o, "").toLocaleLowerCase(),
		this._longMonthsParse[i] = this.months(o, "").toLocaleLowerCase();
		return n ? "MMM" === t ? -1 !== (r = hn.call(this._shortMonthsParse, a)) ? r: null: -1 !== (r = hn.call(this._longMonthsParse, a)) ? r: null: "MMM" === t ? -1 !== (r = hn.call(this._shortMonthsParse, a)) ? r: -1 !== (r = hn.call(this._longMonthsParse, a)) ? r: null: -1 !== (r = hn.call(this._longMonthsParse, a)) ? r: -1 !== (r = hn.call(this._shortMonthsParse, a)) ? r: null
	}
	function Z(e, t) {
		var n;
		if (!e.isValid()) return e;
		if ("string" == typeof t) if (/^\d+$/.test(t)) t = b(t);
		else if (t = e.localeData().monthsParse(t), !o(t)) return e;
		return n = Math.min(e.date(), V(e.year(), t)),
		e._d["set" + (e._isUTC ? "UTC": "") + "Month"](t, n),
		e
	}
	function X(t) {
		return null != t ? (Z(this, t), e.updateOffset(this, !0), this) : E(this, "Month")
	}
	function Q() {
		function e(e, t) {
			return t.length - e.length
		}
		var t, n, i = [],
		r = [],
		o = [];
		for (t = 0; t < 12; t++) n = u([2e3, t]),
		i.push(this.monthsShort(n, "")),
		r.push(this.months(n, "")),
		o.push(this.months(n, "")),
		o.push(this.monthsShort(n, ""));
		for (i.sort(e), r.sort(e), o.sort(e), t = 0; t < 12; t++) i[t] = R(i[t]),
		r[t] = R(r[t]);
		for (t = 0; t < 24; t++) o[t] = R(o[t]);
		this._monthsRegex = new RegExp("^(" + o.join("|") + ")", "i"),
		this._monthsShortRegex = this._monthsRegex,
		this._monthsStrictRegex = new RegExp("^(" + r.join("|") + ")", "i"),
		this._monthsShortStrictRegex = new RegExp("^(" + i.join("|") + ")", "i")
	}
	function J(e) {
		return ee(e) ? 366 : 365
	}
	function ee(e) {
		return e % 4 == 0 && e % 100 != 0 || e % 400 == 0
	}
	function te(e, t, n, i, r, o, a) {
		var s = new Date(e, t, n, i, r, o, a);
		return e < 100 && e >= 0 && isFinite(s.getFullYear()) && s.setFullYear(e),
		s
	}
	function ne(e) {
		var t = new Date(Date.UTC.apply(null, arguments));
		return e < 100 && e >= 0 && isFinite(t.getUTCFullYear()) && t.setUTCFullYear(e),
		t
	}
	function ie(e, t, n) {
		var i = 7 + t - n;
		return - ((7 + ne(e, 0, i).getUTCDay() - t) % 7) + i - 1
	}
	function re(e, t, n, i, r) {
		var o, a, s = 1 + 7 * (t - 1) + (7 + n - i) % 7 + ie(e, i, r);
		return s <= 0 ? (o = e - 1, a = J(o) + s) : s > J(e) ? (o = e + 1, a = s - J(e)) : (o = e, a = s),
		{
			year: o,
			dayOfYear: a
		}
	}
	function oe(e, t, n) {
		var i, r, o = ie(e.year(), t, n),
		a = Math.floor((e.dayOfYear() - o - 1) / 7) + 1;
		return a < 1 ? (r = e.year() - 1, i = a + ae(r, t, n)) : a > ae(e.year(), t, n) ? (i = a - ae(e.year(), t, n), r = e.year() + 1) : (r = e.year(), i = a),
		{
			week: i,
			year: r
		}
	}
	function ae(e, t, n) {
		var i = ie(e, t, n),
		r = ie(e + 1, t, n);
		return (J(e) - i + r) / 7
	}
	function se(e, t) {
		return "string" != typeof e ? e: isNaN(e) ? "number" == typeof(e = t.weekdaysParse(e)) ? e: null: parseInt(e, 10)
	}
	function le(e, t) {
		return "string" == typeof e ? t.weekdaysParse(e) % 7 || 7 : isNaN(e) ? null: e
	}
	function ce(e, t, n) {
		var i, r, o, a = e.toLocaleLowerCase();
		if (!this._weekdaysParse) for (this._weekdaysParse = [], this._shortWeekdaysParse = [], this._minWeekdaysParse = [], i = 0; i < 7; ++i) o = u([2e3, 1]).day(i),
		this._minWeekdaysParse[i] = this.weekdaysMin(o, "").toLocaleLowerCase(),
		this._shortWeekdaysParse[i] = this.weekdaysShort(o, "").toLocaleLowerCase(),
		this._weekdaysParse[i] = this.weekdays(o, "").toLocaleLowerCase();
		return n ? "dddd" === t ? -1 !== (r = hn.call(this._weekdaysParse, a)) ? r: null: "ddd" === t ? -1 !== (r = hn.call(this._shortWeekdaysParse, a)) ? r: null: -1 !== (r = hn.call(this._minWeekdaysParse, a)) ? r: null: "dddd" === t ? -1 !== (r = hn.call(this._weekdaysParse, a)) ? r: -1 !== (r = hn.call(this._shortWeekdaysParse, a)) ? r: -1 !== (r = hn.call(this._minWeekdaysParse, a)) ? r: null: "ddd" === t ? -1 !== (r = hn.call(this._shortWeekdaysParse, a)) ? r: -1 !== (r = hn.call(this._weekdaysParse, a)) ? r: -1 !== (r = hn.call(this._minWeekdaysParse, a)) ? r: null: -1 !== (r = hn.call(this._minWeekdaysParse, a)) ? r: -1 !== (r = hn.call(this._weekdaysParse, a)) ? r: -1 !== (r = hn.call(this._shortWeekdaysParse, a)) ? r: null
	}
	function ue() {
		function e(e, t) {
			return t.length - e.length
		}
		var t, n, i, r, o, a = [],
		s = [],
		l = [],
		c = [];
		for (t = 0; t < 7; t++) n = u([2e3, 1]).day(t),
		i = this.weekdaysMin(n, ""),
		r = this.weekdaysShort(n, ""),
		o = this.weekdays(n, ""),
		a.push(i),
		s.push(r),
		l.push(o),
		c.push(i),
		c.push(r),
		c.push(o);
		for (a.sort(e), s.sort(e), l.sort(e), c.sort(e), t = 0; t < 7; t++) s[t] = R(s[t]),
		l[t] = R(l[t]),
		c[t] = R(c[t]);
		this._weekdaysRegex = new RegExp("^(" + c.join("|") + ")", "i"),
		this._weekdaysShortRegex = this._weekdaysRegex,
		this._weekdaysMinRegex = this._weekdaysRegex,
		this._weekdaysStrictRegex = new RegExp("^(" + l.join("|") + ")", "i"),
		this._weekdaysShortStrictRegex = new RegExp("^(" + s.join("|") + ")", "i"),
		this._weekdaysMinStrictRegex = new RegExp("^(" + a.join("|") + ")", "i")
	}
	function de() {
		return this.hours() % 12 || 12
	}
	function he(e, t) {
		j(e, 0, 0,
		function() {
			return this.localeData().meridiem(this.hours(), this.minutes(), t)
		})
	}
	function fe(e, t) {
		return t._meridiemParse
	}
	function pe(e) {
		return e ? e.toLowerCase().replace("_", "-") : e
	}
	function me(e) {
		for (var t, n, i, r, o = 0; o < e.length;) {
			for (t = (r = pe(e[o]).split("-")).length, n = (n = pe(e[o + 1])) ? n.split("-") : null; t > 0;) {
				if (i = ge(r.slice(0, t).join("-"))) return i;
				if (n && n.length >= t && w(r, n, !0) >= t - 1) break;
				t--
			}
			o++
		}
		return null
	}
	function ge(e) {
		var t = null;
		if (!On[e] && "undefined" != typeof module && module && module.exports) try {
			t = Mn._abbr,
			require("./locale/" + e),
			ve(t)
		} catch(e) {}
		return On[e]
	}
	function ve(e, t) {
		var n;
		return e && (n = r(t) ? be(e) : ye(e, t)) && (Mn = n),
		Mn._abbr
	}
	function ye(e, t) {
		if (null !== t) {
			var n = Ln;
			if (t.abbr = e, null != On[e]) _("defineLocaleOverride", "use moment.updateLocale(localeName, config) to change an existing locale. moment.defineLocale(localeName, config) should only be used for creating a new locale See http://momentjs.com/guides/#/warnings/define-locale/ for more info."),
			n = On[e]._config;
			else if (null != t.parentLocale) {
				if (null == On[t.parentLocale]) return Nn[t.parentLocale] || (Nn[t.parentLocale] = []),
				Nn[t.parentLocale].push({
					name: e,
					config: t
				}),
				null;
				n = On[t.parentLocale]._config
			}
			return On[e] = new M(S(n, t)),
			Nn[e] && Nn[e].forEach(function(e) {
				ye(e.name, e.config)
			}),
			ve(e),
			On[e]
		}
		return delete On[e],
		null
	}
	function be(e) {
		var n;
		if (e && e._locale && e._locale._abbr && (e = e._locale._abbr), !e) return Mn;
		if (!t(e)) {
			if (n = ge(e)) return n;
			e = [e]
		}
		return me(e)
	}
	function we(e) {
		var t, n = e._a;
		return n && -2 === h(e).overflow && (t = n[rn] < 0 || n[rn] > 11 ? rn: n[on] < 1 || n[on] > V(n[nn], n[rn]) ? on: n[an] < 0 || n[an] > 24 || 24 === n[an] && (0 !== n[sn] || 0 !== n[ln] || 0 !== n[cn]) ? an: n[sn] < 0 || n[sn] > 59 ? sn: n[ln] < 0 || n[ln] > 59 ? ln: n[cn] < 0 || n[cn] > 999 ? cn: -1, h(e)._overflowDayOfYear && (t < nn || t > on) && (t = on), h(e)._overflowWeeks && -1 === t && (t = un), h(e)._overflowWeekday && -1 === t && (t = dn), h(e).overflow = t),
		e
	}
	function ke(e) {
		var t, n, i, r, o, a, s = e._i,
		l = An.exec(s) || En.exec(s);
		if (l) {
			for (h(e).iso = !0, t = 0, n = qn.length; t < n; t++) if (qn[t][1].exec(l[1])) {
				r = qn[t][0],
				i = !1 !== qn[t][2];
				break
			}
			if (null == r) return void(e._isValid = !1);
			if (l[3]) {
				for (t = 0, n = jn.length; t < n; t++) if (jn[t][1].exec(l[3])) {
					o = (l[2] || " ") + jn[t][0];
					break
				}
				if (null == o) return void(e._isValid = !1)
			}
			if (!i && null != o) return void(e._isValid = !1);
			if (l[4]) {
				if (!$n.exec(l[4])) return void(e._isValid = !1);
				a = "Z"
			}
			e._f = r + (o || "") + (a || ""),
			De(e)
		} else e._isValid = !1
	}
	function xe(e) {
		var t, n, i, r, o, a, s, l, c = {
			" GMT": " +0000",
			" EDT": " -0400",
			" EST": " -0500",
			" CDT": " -0500",
			" CST": " -0600",
			" MDT": " -0600",
			" MST": " -0700",
			" PDT": " -0700",
			" PST": " -0800"
		};
		if (t = e._i.replace(/\([^\)]*\)|[\n\t]/g, " ").replace(/(\s\s+)/g, " ").replace(/^\s|\s$/g, ""), n = In.exec(t)) {
			if (i = n[1] ? "ddd" + (5 === n[1].length ? ", ": " ") : "", r = "D MMM " + (n[2].length > 10 ? "YYYY ": "YY "), o = "HH:mm" + (n[4] ? ":ss": ""), n[1]) {
				var u = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"][new Date(n[2]).getDay()];
				if (n[1].substr(0, 3) !== u) return h(e).weekdayMismatch = !0,
				void(e._isValid = !1)
			}
			switch (n[5].length) {
			case 2:
				0 === l ? s = " +0000": (l = "YXWVUTSRQPONZABCDEFGHIKLM".indexOf(n[5][1].toUpperCase()) - 12, s = (l < 0 ? " -": " +") + ("" + l).replace(/^-?/, "0").match(/..$/)[0] + "00");
				break;
			case 4:
				s = c[n[5]];
				break;
			default:
				s = c[" GMT"]
			}
			n[5] = s,
			e._i = n.splice(1).join(""),
			a = " ZZ",
			e._f = i + r + o + a,
			De(e),
			h(e).rfc2822 = !0
		} else e._isValid = !1
	}
	function _e(t) {
		var n = Pn.exec(t._i);
		return null !== n ? void(t._d = new Date( + n[1])) : (ke(t), void(!1 === t._isValid && (delete t._isValid, xe(t), !1 === t._isValid && (delete t._isValid, e.createFromInputFallback(t)))))
	}
	function Ce(e, t, n) {
		return null != e ? e: null != t ? t: n
	}
	function Se(t) {
		var n = new Date(e.now());
		return t._useUTC ? [n.getUTCFullYear(), n.getUTCMonth(), n.getUTCDate()] : [n.getFullYear(), n.getMonth(), n.getDate()]
	}
	function Me(e) {
		var t, n, i, r, o = [];
		if (!e._d) {
			for (i = Se(e), e._w && null == e._a[on] && null == e._a[rn] && Te(e), null != e._dayOfYear && (r = Ce(e._a[nn], i[nn]), (e._dayOfYear > J(r) || 0 === e._dayOfYear) && (h(e)._overflowDayOfYear = !0), n = ne(r, 0, e._dayOfYear), e._a[rn] = n.getUTCMonth(), e._a[on] = n.getUTCDate()), t = 0; t < 3 && null == e._a[t]; ++t) e._a[t] = o[t] = i[t];
			for (; t < 7; t++) e._a[t] = o[t] = null == e._a[t] ? 2 === t ? 1 : 0 : e._a[t];
			24 === e._a[an] && 0 === e._a[sn] && 0 === e._a[ln] && 0 === e._a[cn] && (e._nextDay = !0, e._a[an] = 0),
			e._d = (e._useUTC ? ne: te).apply(null, o),
			null != e._tzm && e._d.setUTCMinutes(e._d.getUTCMinutes() - e._tzm),
			e._nextDay && (e._a[an] = 24)
		}
	}
	function Te(e) {
		var t, n, i, r, o, a, s, l;
		if (null != (t = e._w).GG || null != t.W || null != t.E) o = 1,
		a = 4,
		n = Ce(t.GG, e._a[nn], oe(je(), 1, 4).year),
		i = Ce(t.W, 1),
		((r = Ce(t.E, 1)) < 1 || r > 7) && (l = !0);
		else {
			o = e._locale._week.dow,
			a = e._locale._week.doy;
			var c = oe(je(), o, a);
			n = Ce(t.gg, e._a[nn], c.year),
			i = Ce(t.w, c.week),
			null != t.d ? ((r = t.d) < 0 || r > 6) && (l = !0) : null != t.e ? (r = t.e + o, (t.e < 0 || t.e > 6) && (l = !0)) : r = o
		}
		i < 1 || i > ae(n, o, a) ? h(e)._overflowWeeks = !0 : null != l ? h(e)._overflowWeekday = !0 : (s = re(n, i, r, o, a), e._a[nn] = s.year, e._dayOfYear = s.dayOfYear)
	}
	function De(t) {
		if (t._f !== e.ISO_8601) if (t._f !== e.RFC_2822) {
			t._a = [],
			h(t).empty = !0;
			var n, i, r, o, a, s = "" + t._i,
			l = s.length,
			c = 0;
			for (r = W(t._f, t._locale).match(Et) || [], n = 0; n < r.length; n++) o = r[n],
			(i = (s.match(F(o, t)) || [])[0]) && ((a = s.substr(0, s.indexOf(i))).length > 0 && h(t).unusedInput.push(a), s = s.slice(s.indexOf(i) + i.length), c += i.length),
			jt[o] ? (i ? h(t).empty = !1 : h(t).unusedTokens.push(o), G(o, i, t)) : t._strict && !i && h(t).unusedTokens.push(o);
			h(t).charsLeftOver = l - c,
			s.length > 0 && h(t).unusedInput.push(s),
			t._a[an] <= 12 && !0 === h(t).bigHour && t._a[an] > 0 && (h(t).bigHour = void 0),
			h(t).parsedDateParts = t._a.slice(0),
			h(t).meridiem = t._meridiem,
			t._a[an] = Le(t._locale, t._a[an], t._meridiem),
			Me(t),
			we(t)
		} else xe(t);
		else ke(t)
	}
	function Le(e, t, n) {
		var i;
		return null == n ? t: null != e.meridiemHour ? e.meridiemHour(t, n) : null != e.isPM ? ((i = e.isPM(n)) && t < 12 && (t += 12), i || 12 !== t || (t = 0), t) : t
	}
	function Oe(e) {
		var t, n, i, r, o;
		if (0 === e._f.length) return h(e).invalidFormat = !0,
		void(e._d = new Date(NaN));
		for (r = 0; r < e._f.length; r++) o = 0,
		t = m({},
		e),
		null != e._useUTC && (t._useUTC = e._useUTC),
		t._f = e._f[r],
		De(t),
		f(t) && (o += h(t).charsLeftOver, o += 10 * h(t).unusedTokens.length, h(t).score = o, (null == i || o < i) && (i = o, n = t));
		c(e, n || t)
	}
	function Ne(e) {
		if (!e._d) {
			var t = L(e._i);
			e._a = s([t.year, t.month, t.day || t.date, t.hour, t.minute, t.second, t.millisecond],
			function(e) {
				return e && parseInt(e, 10)
			}),
			Me(e)
		}
	}
	function Ae(e) {
		var t = new g(we(Ee(e)));
		return t._nextDay && (t.add(1, "d"), t._nextDay = void 0),
		t
	}
	function Ee(e) {
		var n = e._i,
		i = e._f;
		return e._locale = e._locale || be(e._l),
		null === n || void 0 === i && "" === n ? p({
			nullInput: !0
		}) : ("string" == typeof n && (e._i = n = e._locale.preparse(n)), v(n) ? new g(we(n)) : (a(n) ? e._d = n: t(i) ? Oe(e) : i ? De(e) : $e(e), f(e) || (e._d = null), e))
	}
	function $e(i) {
		var l = i._i;
		r(l) ? i._d = new Date(e.now()) : a(l) ? i._d = new Date(l.valueOf()) : "string" == typeof l ? _e(i) : t(l) ? (i._a = s(l.slice(0),
		function(e) {
			return parseInt(e, 10)
		}), Me(i)) : n(l) ? Ne(i) : o(l) ? i._d = new Date(l) : e.createFromInputFallback(i)
	}
	function qe(e, r, o, a, s) {
		var l = {};
		return ! 0 !== o && !1 !== o || (a = o, o = void 0),
		(n(e) && i(e) || t(e) && 0 === e.length) && (e = void 0),
		l._isAMomentObject = !0,
		l._useUTC = l._isUTC = s,
		l._l = o,
		l._i = e,
		l._f = r,
		l._strict = a,
		Ae(l)
	}
	function je(e, t, n, i) {
		return qe(e, t, n, i, !1)
	}
	function Pe(e, n) {
		var i, r;
		if (1 === n.length && t(n[0]) && (n = n[0]), !n.length) return je();
		for (i = n[0], r = 1; r < n.length; ++r) n[r].isValid() && !n[r][e](i) || (i = n[r]);
		return i
	}
	function Ie(e) {
		for (var t in e) if ( - 1 === Hn.indexOf(t) || null != e[t] && isNaN(e[t])) return ! 1;
		for (var n = !1,
		i = 0; i < Hn.length; ++i) if (e[Hn[i]]) {
			if (n) return ! 1;
			parseFloat(e[Hn[i]]) !== b(e[Hn[i]]) && (n = !0)
		}
		return ! 0
	}
	function ze(e) {
		var t = L(e),
		n = t.year || 0,
		i = t.quarter || 0,
		r = t.month || 0,
		o = t.week || 0,
		a = t.day || 0,
		s = t.hour || 0,
		l = t.minute || 0,
		c = t.second || 0,
		u = t.millisecond || 0;
		this._isValid = Ie(t),
		this._milliseconds = +u + 1e3 * c + 6e4 * l + 1e3 * s * 60 * 60,
		this._days = +a + 7 * o,
		this._months = +r + 3 * i + 12 * n,
		this._data = {},
		this._locale = be(),
		this._bubble()
	}
	function We(e) {
		return e instanceof ze
	}
	function He(e) {
		return e < 0 ? -1 * Math.round( - 1 * e) : Math.round(e)
	}
	function Fe(e, t) {
		j(e, 0, 0,
		function() {
			var e = this.utcOffset(),
			n = "+";
			return e < 0 && (e = -e, n = "-"),
			n + q(~~ (e / 60), 2) + t + q(~~e % 60, 2)
		})
	}
	function Ye(e, t) {
		var n = (t || "").match(e);
		if (null === n) return null;
		var i = ((n[n.length - 1] || []) + "").match(Fn) || ["-", 0, 0],
		r = 60 * i[1] + b(i[2]);
		return 0 === r ? 0 : "+" === i[0] ? r: -r
	}
	function Re(t, n) {
		var i, r;
		return n._isUTC ? (i = n.clone(), r = (v(t) || a(t) ? t.valueOf() : je(t).valueOf()) - i.valueOf(), i._d.setTime(i._d.valueOf() + r), e.updateOffset(i, !1), i) : je(t).local()
	}
	function Ue(e) {
		return 15 * -Math.round(e._d.getTimezoneOffset() / 15)
	}
	function Be() {
		return !! this.isValid() && this._isUTC && 0 === this._offset
	}
	function Ge(e, t) {
		var n, i, r, a = e,
		s = null;
		return We(e) ? a = {
			ms: e._milliseconds,
			d: e._days,
			M: e._months
		}: o(e) ? (a = {},
		t ? a[t] = e: a.milliseconds = e) : (s = Yn.exec(e)) ? (n = "-" === s[1] ? -1 : 1, a = {
			y: 0,
			d: b(s[on]) * n,
			h: b(s[an]) * n,
			m: b(s[sn]) * n,
			s: b(s[ln]) * n,
			ms: b(He(1e3 * s[cn])) * n
		}) : (s = Rn.exec(e)) ? (n = "-" === s[1] ? -1 : 1, a = {
			y: Ve(s[2], n),
			M: Ve(s[3], n),
			w: Ve(s[4], n),
			d: Ve(s[5], n),
			h: Ve(s[6], n),
			m: Ve(s[7], n),
			s: Ve(s[8], n)
		}) : null == a ? a = {}: "object" == typeof a && ("from" in a || "to" in a) && (r = Ze(je(a.from), je(a.to)), a = {},
		a.ms = r.milliseconds, a.M = r.months),
		i = new ze(a),
		We(e) && l(e, "_locale") && (i._locale = e._locale),
		i
	}
	function Ve(e, t) {
		var n = e && parseFloat(e.replace(",", "."));
		return (isNaN(n) ? 0 : n) * t
	}
	function Ke(e, t) {
		var n = {
			milliseconds: 0,
			months: 0
		};
		return n.months = t.month() - e.month() + 12 * (t.year() - e.year()),
		e.clone().add(n.months, "M").isAfter(t) && --n.months,
		n.milliseconds = +t - +e.clone().add(n.months, "M"),
		n
	}
	function Ze(e, t) {
		var n;
		return e.isValid() && t.isValid() ? (t = Re(t, e), e.isBefore(t) ? n = Ke(e, t) : (n = Ke(t, e), n.milliseconds = -n.milliseconds, n.months = -n.months), n) : {
			milliseconds: 0,
			months: 0
		}
	}
	function Xe(e, t) {
		return function(n, i) {
			var r, o;
			return null === i || isNaN( + i) || (_(t, "moment()." + t + "(period, number) is deprecated. Please use moment()." + t + "(number, period). See http://momentjs.com/guides/#/warnings/add-inverted-param/ for more info."), o = n, n = i, i = o),
			n = "string" == typeof n ? +n: n,
			r = Ge(n, i),
			Qe(this, r, e),
			this
		}
	}
	function Qe(t, n, i, r) {
		var o = n._milliseconds,
		a = He(n._days),
		s = He(n._months);
		t.isValid() && (r = null == r || r, o && t._d.setTime(t._d.valueOf() + o * i), a && $(t, "Date", E(t, "Date") + a * i), s && Z(t, E(t, "Month") + s * i), r && e.updateOffset(t, a || s))
	}
	function Je(e, t) {
		var n, i, r = 12 * (t.year() - e.year()) + (t.month() - e.month()),
		o = e.clone().add(r, "months");
		return t - o < 0 ? (n = e.clone().add(r - 1, "months"), i = (t - o) / (o - n)) : (n = e.clone().add(r + 1, "months"), i = (t - o) / (n - o)),
		-(r + i) || 0
	}
	function et(e) {
		var t;
		return void 0 === e ? this._locale._abbr: (null != (t = be(e)) && (this._locale = t), this)
	}
	function tt() {
		return this._locale
	}
	function nt(e, t) {
		j(0, [e, e.length], 0, t)
	}
	function it(e, t, n, i, r) {
		var o;
		return null == e ? oe(this, i, r).year: (o = ae(e, i, r), t > o && (t = o), rt.call(this, e, t, n, i, r))
	}
	function rt(e, t, n, i, r) {
		var o = re(e, t, n, i, r),
		a = ne(o.year, 0, o.dayOfYear);
		return this.year(a.getUTCFullYear()),
		this.month(a.getUTCMonth()),
		this.date(a.getUTCDate()),
		this
	}
	function ot(e) {
		return e
	}
	function at(e, t, n, i) {
		var r = be(),
		o = u().set(i, t);
		return r[n](o, e)
	}
	function st(e, t, n) {
		if (o(e) && (t = e, e = void 0), e = e || "", null != t) return at(e, t, n, "month");
		var i, r = [];
		for (i = 0; i < 12; i++) r[i] = at(e, i, n, "month");
		return r
	}
	function lt(e, t, n, i) {
		"boolean" == typeof e ? (o(t) && (n = t, t = void 0), t = t || "") : (t = e, n = t, e = !1, o(t) && (n = t, t = void 0), t = t || "");
		var r = be(),
		a = e ? r._week.dow: 0;
		if (null != n) return at(t, (n + a) % 7, i, "day");
		var s, l = [];
		for (s = 0; s < 7; s++) l[s] = at(t, (s + a) % 7, i, "day");
		return l
	}
	function ct(e, t, n, i) {
		var r = Ge(t, n);
		return e._milliseconds += i * r._milliseconds,
		e._days += i * r._days,
		e._months += i * r._months,
		e._bubble()
	}
	function ut(e) {
		return e < 0 ? Math.floor(e) : Math.ceil(e)
	}
	function dt(e) {
		return 4800 * e / 146097
	}
	function ht(e) {
		return 146097 * e / 4800
	}
	function ft(e) {
		return function() {
			return this.as(e)
		}
	}
	function pt(e) {
		return function() {
			return this.isValid() ? this._data[e] : NaN
		}
	}
	function mt(e, t, n, i, r) {
		return r.relativeTime(t || 1, !!n, e, i)
	}
	function gt(e, t, n) {
		var i = Ge(e).abs(),
		r = vi(i.as("s")),
		o = vi(i.as("m")),
		a = vi(i.as("h")),
		s = vi(i.as("d")),
		l = vi(i.as("M")),
		c = vi(i.as("y")),
		u = r <= yi.ss && ["s", r] || r < yi.s && ["ss", r] || o <= 1 && ["m"] || o < yi.m && ["mm", o] || a <= 1 && ["h"] || a < yi.h && ["hh", a] || s <= 1 && ["d"] || s < yi.d && ["dd", s] || l <= 1 && ["M"] || l < yi.M && ["MM", l] || c <= 1 && ["y"] || ["yy", c];
		return u[2] = t,
		u[3] = +e > 0,
		u[4] = n,
		mt.apply(null, u)
	}
	function vt() {
		if (!this.isValid()) return this.localeData().invalidDate();
		var e, t, n, i = bi(this._milliseconds) / 1e3,
		r = bi(this._days),
		o = bi(this._months);
		t = y((e = y(i / 60)) / 60),
		i %= 60,
		e %= 60;
		var a = n = y(o / 12),
		s = o %= 12,
		l = r,
		c = t,
		u = e,
		d = i,
		h = this.asSeconds();
		return h ? (h < 0 ? "-": "") + "P" + (a ? a + "Y": "") + (s ? s + "M": "") + (l ? l + "D": "") + (c || u || d ? "T": "") + (c ? c + "H": "") + (u ? u + "M": "") + (d ? d + "S": "") : "P0D"
	}
	var yt, bt, wt = bt = Array.prototype.some ? Array.prototype.some: function(e) {
		for (var t = Object(this), n = t.length >>> 0, i = 0; i < n; i++) if (i in t && e.call(this, t[i], i, t)) return ! 0;
		return ! 1
	},
	kt = e.momentProperties = [],
	xt = !1,
	_t = {};
	e.suppressDeprecationWarnings = !1,
	e.deprecationHandler = null;
	var Ct, St, Mt = Ct = Object.keys ? Object.keys: function(e) {
		var t, n = [];
		for (t in e) l(e, t) && n.push(t);
		return n
	},
	Tt = {
		sameDay: "[Today at] LT",
		nextDay: "[Tomorrow at] LT",
		nextWeek: "dddd [at] LT",
		lastDay: "[Yesterday at] LT",
		lastWeek: "[Last] dddd [at] LT",
		sameElse: "L"
	},
	Dt = {
		LTS: "h:mm:ss A",
		LT: "h:mm A",
		L: "MM/DD/YYYY",
		LL: "MMMM D, YYYY",
		LLL: "MMMM D, YYYY h:mm A",
		LLLL: "dddd, MMMM D, YYYY h:mm A"
	},
	Lt = /\d{1,2}/,
	Ot = {
		future: "in %s",
		past: "%s ago",
		s: "a few seconds",
		ss: "%d seconds",
		m: "a minute",
		mm: "%d minutes",
		h: "an hour",
		hh: "%d hours",
		d: "a day",
		dd: "%d days",
		M: "a month",
		MM: "%d months",
		y: "a year",
		yy: "%d years"
	},
	Nt = {},
	At = {},
	Et = /(\[[^\[]*\])|(\\)?([Hh]mm(ss)?|Mo|MM?M?M?|Do|DDDo|DD?D?D?|ddd?d?|do?|w[o|w]?|W[o|W]?|Qo?|YYYYYY|YYYYY|YYYY|YY|gg(ggg?)?|GG(GGG?)?|e|E|a|A|hh?|HH?|kk?|mm?|ss?|S{1,9}|x|X|zz?|ZZ?|.)/g,
	$t = /(\[[^\[]*\])|(\\)?(LTS|LT|LL?L?L?|l{1,4})/g,
	qt = {},
	jt = {},
	Pt = /\d/,
	It = /\d\d/,
	zt = /\d{3}/,
	Wt = /\d{4}/,
	Ht = /[+-]?\d{6}/,
	Ft = /\d\d?/,
	Yt = /\d\d\d\d?/,
	Rt = /\d\d\d\d\d\d?/,
	Ut = /\d{1,3}/,
	Bt = /\d{1,4}/,
	Gt = /[+-]?\d{1,6}/,
	Vt = /\d+/,
	Kt = /[+-]?\d+/,
	Zt = /Z|[+-]\d\d:?\d\d/gi,
	Xt = /Z|[+-]\d\d(?::?\d\d)?/gi,
	Qt = /[+-]?\d+(\.\d{1,3})?/,
	Jt = /[0-9]*['a-z\u00A0-\u05FF\u0700-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF]+|[\u0600-\u06FF\/]+(\s*?[\u0600-\u06FF]+){1,2}/i,
	en = {},
	tn = {},
	nn = 0,
	rn = 1,
	on = 2,
	an = 3,
	sn = 4,
	ln = 5,
	cn = 6,
	un = 7,
	dn = 8,
	hn = St = Array.prototype.indexOf ? Array.prototype.indexOf: function(e) {
		var t;
		for (t = 0; t < this.length; ++t) if (this[t] === e) return t;
		return - 1
	};
	j("M", ["MM", 2], "Mo",
	function() {
		return this.month() + 1
	}),
	j("MMM", 0, 0,
	function(e) {
		return this.localeData().monthsShort(this, e)
	}),
	j("MMMM", 0, 0,
	function(e) {
		return this.localeData().months(this, e)
	}),
	T("month", "M"),
	O("month", 8),
	H("M", Ft),
	H("MM", Ft, It),
	H("MMM",
	function(e, t) {
		return t.monthsShortRegex(e)
	}),
	H("MMMM",
	function(e, t) {
		return t.monthsRegex(e)
	}),
	U(["M", "MM"],
	function(e, t) {
		t[rn] = b(e) - 1
	}),
	U(["MMM", "MMMM"],
	function(e, t, n, i) {
		var r = n._locale.monthsParse(e, i, n._strict);
		null != r ? t[rn] = r: h(n).invalidMonth = e
	});
	var fn = /D[oD]?(\[[^\[\]]*\]|\s)+MMMM?/,
	pn = "January_February_March_April_May_June_July_August_September_October_November_December".split("_"),
	mn = "Jan_Feb_Mar_Apr_May_Jun_Jul_Aug_Sep_Oct_Nov_Dec".split("_"),
	gn = Jt,
	vn = Jt;
	j("Y", 0, 0,
	function() {
		var e = this.year();
		return e <= 9999 ? "" + e: "+" + e
	}),
	j(0, ["YY", 2], 0,
	function() {
		return this.year() % 100
	}),
	j(0, ["YYYY", 4], 0, "year"),
	j(0, ["YYYYY", 5], 0, "year"),
	j(0, ["YYYYYY", 6, !0], 0, "year"),
	T("year", "y"),
	O("year", 1),
	H("Y", Kt),
	H("YY", Ft, It),
	H("YYYY", Bt, Wt),
	H("YYYYY", Gt, Ht),
	H("YYYYYY", Gt, Ht),
	U(["YYYYY", "YYYYYY"], nn),
	U("YYYY",
	function(t, n) {
		n[nn] = 2 === t.length ? e.parseTwoDigitYear(t) : b(t)
	}),
	U("YY",
	function(t, n) {
		n[nn] = e.parseTwoDigitYear(t)
	}),
	U("Y",
	function(e, t) {
		t[nn] = parseInt(e, 10)
	}),
	e.parseTwoDigitYear = function(e) {
		return b(e) + (b(e) > 68 ? 1900 : 2e3)
	};
	var yn = A("FullYear", !0);
	j("w", ["ww", 2], "wo", "week"),
	j("W", ["WW", 2], "Wo", "isoWeek"),
	T("week", "w"),
	T("isoWeek", "W"),
	O("week", 5),
	O("isoWeek", 5),
	H("w", Ft),
	H("ww", Ft, It),
	H("W", Ft),
	H("WW", Ft, It),
	B(["w", "ww", "W", "WW"],
	function(e, t, n, i) {
		t[i.substr(0, 1)] = b(e)
	});
	var bn = {
		dow: 0,
		doy: 6
	};
	j("d", 0, "do", "day"),
	j("dd", 0, 0,
	function(e) {
		return this.localeData().weekdaysMin(this, e)
	}),
	j("ddd", 0, 0,
	function(e) {
		return this.localeData().weekdaysShort(this, e)
	}),
	j("dddd", 0, 0,
	function(e) {
		return this.localeData().weekdays(this, e)
	}),
	j("e", 0, 0, "weekday"),
	j("E", 0, 0, "isoWeekday"),
	T("day", "d"),
	T("weekday", "e"),
	T("isoWeekday", "E"),
	O("day", 11),
	O("weekday", 11),
	O("isoWeekday", 11),
	H("d", Ft),
	H("e", Ft),
	H("E", Ft),
	H("dd",
	function(e, t) {
		return t.weekdaysMinRegex(e)
	}),
	H("ddd",
	function(e, t) {
		return t.weekdaysShortRegex(e)
	}),
	H("dddd",
	function(e, t) {
		return t.weekdaysRegex(e)
	}),
	B(["dd", "ddd", "dddd"],
	function(e, t, n, i) {
		var r = n._locale.weekdaysParse(e, i, n._strict);
		null != r ? t.d = r: h(n).invalidWeekday = e
	}),
	B(["d", "e", "E"],
	function(e, t, n, i) {
		t[i] = b(e)
	});
	var wn = "Sunday_Monday_Tuesday_Wednesday_Thursday_Friday_Saturday".split("_"),
	kn = "Sun_Mon_Tue_Wed_Thu_Fri_Sat".split("_"),
	xn = "Su_Mo_Tu_We_Th_Fr_Sa".split("_"),
	_n = Jt,
	Cn = Jt,
	Sn = Jt;
	j("H", ["HH", 2], 0, "hour"),
	j("h", ["hh", 2], 0, de),
	j("k", ["kk", 2], 0,
	function() {
		return this.hours() || 24
	}),
	j("hmm", 0, 0,
	function() {
		return "" + de.apply(this) + q(this.minutes(), 2)
	}),
	j("hmmss", 0, 0,
	function() {
		return "" + de.apply(this) + q(this.minutes(), 2) + q(this.seconds(), 2)
	}),
	j("Hmm", 0, 0,
	function() {
		return "" + this.hours() + q(this.minutes(), 2)
	}),
	j("Hmmss", 0, 0,
	function() {
		return "" + this.hours() + q(this.minutes(), 2) + q(this.seconds(), 2)
	}),
	he("a", !0),
	he("A", !1),
	T("hour", "h"),
	O("hour", 13),
	H("a", fe),
	H("A", fe),
	H("H", Ft),
	H("h", Ft),
	H("k", Ft),
	H("HH", Ft, It),
	H("hh", Ft, It),
	H("kk", Ft, It),
	H("hmm", Yt),
	H("hmmss", Rt),
	H("Hmm", Yt),
	H("Hmmss", Rt),
	U(["H", "HH"], an),
	U(["k", "kk"],
	function(e, t, n) {
		var i = b(e);
		t[an] = 24 === i ? 0 : i
	}),
	U(["a", "A"],
	function(e, t, n) {
		n._isPm = n._locale.isPM(e),
		n._meridiem = e
	}),
	U(["h", "hh"],
	function(e, t, n) {
		t[an] = b(e),
		h(n).bigHour = !0
	}),
	U("hmm",
	function(e, t, n) {
		var i = e.length - 2;
		t[an] = b(e.substr(0, i)),
		t[sn] = b(e.substr(i)),
		h(n).bigHour = !0
	}),
	U("hmmss",
	function(e, t, n) {
		var i = e.length - 4,
		r = e.length - 2;
		t[an] = b(e.substr(0, i)),
		t[sn] = b(e.substr(i, 2)),
		t[ln] = b(e.substr(r)),
		h(n).bigHour = !0
	}),
	U("Hmm",
	function(e, t, n) {
		var i = e.length - 2;
		t[an] = b(e.substr(0, i)),
		t[sn] = b(e.substr(i))
	}),
	U("Hmmss",
	function(e, t, n) {
		var i = e.length - 4,
		r = e.length - 2;
		t[an] = b(e.substr(0, i)),
		t[sn] = b(e.substr(i, 2)),
		t[ln] = b(e.substr(r))
	});
	var Mn, Tn = /[ap]\.?m?\.?/i,
	Dn = A("Hours", !0),
	Ln = {
		calendar: Tt,
		longDateFormat: Dt,
		invalidDate: "Invalid date",
		ordinal: "%d",
		dayOfMonthOrdinalParse: Lt,
		relativeTime: Ot,
		months: pn,
		monthsShort: mn,
		week: bn,
		weekdays: wn,
		weekdaysMin: xn,
		weekdaysShort: kn,
		meridiemParse: Tn
	},
	On = {},
	Nn = {},
	An = /^\s*((?:[+-]\d{6}|\d{4})-(?:\d\d-\d\d|W\d\d-\d|W\d\d|\d\d\d|\d\d))(?:(T| )(\d\d(?::\d\d(?::\d\d(?:[.,]\d+)?)?)?)([\+\-]\d\d(?::?\d\d)?|\s*Z)?)?$/,
	En = /^\s*((?:[+-]\d{6}|\d{4})(?:\d\d\d\d|W\d\d\d|W\d\d|\d\d\d|\d\d))(?:(T| )(\d\d(?:\d\d(?:\d\d(?:[.,]\d+)?)?)?)([\+\-]\d\d(?::?\d\d)?|\s*Z)?)?$/,
	$n = /Z|[+-]\d\d(?::?\d\d)?/,
	qn = [["YYYYYY-MM-DD", /[+-]\d{6}-\d\d-\d\d/], ["YYYY-MM-DD", /\d{4}-\d\d-\d\d/], ["GGGG-[W]WW-E", /\d{4}-W\d\d-\d/], ["GGGG-[W]WW", /\d{4}-W\d\d/, !1], ["YYYY-DDD", /\d{4}-\d{3}/], ["YYYY-MM", /\d{4}-\d\d/, !1], ["YYYYYYMMDD", /[+-]\d{10}/], ["YYYYMMDD", /\d{8}/], ["GGGG[W]WWE", /\d{4}W\d{3}/], ["GGGG[W]WW", /\d{4}W\d{2}/, !1], ["YYYYDDD", /\d{7}/]],
	jn = [["HH:mm:ss.SSSS", /\d\d:\d\d:\d\d\.\d+/], ["HH:mm:ss,SSSS", /\d\d:\d\d:\d\d,\d+/], ["HH:mm:ss", /\d\d:\d\d:\d\d/], ["HH:mm", /\d\d:\d\d/], ["HHmmss.SSSS", /\d\d\d\d\d\d\.\d+/], ["HHmmss,SSSS", /\d\d\d\d\d\d,\d+/], ["HHmmss", /\d\d\d\d\d\d/], ["HHmm", /\d\d\d\d/], ["HH", /\d\d/]],
	Pn = /^\/?Date\((\-?\d+)/i,
	In = /^((?:Mon|Tue|Wed|Thu|Fri|Sat|Sun),?\s)?(\d?\d\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s(?:\d\d)?\d\d\s)(\d\d:\d\d)(\:\d\d)?(\s(?:UT|GMT|[ECMP][SD]T|[A-IK-Za-ik-z]|[+-]\d{4}))$/;
	e.createFromInputFallback = x("value provided is not in a recognized RFC2822 or ISO format. moment construction falls back to js Date(), which is not reliable across all browsers and versions. Non RFC2822/ISO date formats are discouraged and will be removed in an upcoming major release. Please refer to http://momentjs.com/guides/#/warnings/js-date/ for more info.",
	function(e) {
		e._d = new Date(e._i + (e._useUTC ? " UTC": ""))
	}),
	e.ISO_8601 = function() {},
	e.RFC_2822 = function() {};
	var zn = x("moment().min is deprecated, use moment.max instead. http://momentjs.com/guides/#/warnings/min-max/",
	function() {
		var e = je.apply(null, arguments);
		return this.isValid() && e.isValid() ? e < this ? this: e: p()
	}),
	Wn = x("moment().max is deprecated, use moment.min instead. http://momentjs.com/guides/#/warnings/min-max/",
	function() {
		var e = je.apply(null, arguments);
		return this.isValid() && e.isValid() ? e > this ? this: e: p()
	}),
	Hn = ["year", "quarter", "month", "week", "day", "hour", "minute", "second", "millisecond"];
	Fe("Z", ":"),
	Fe("ZZ", ""),
	H("Z", Xt),
	H("ZZ", Xt),
	U(["Z", "ZZ"],
	function(e, t, n) {
		n._useUTC = !0,
		n._tzm = Ye(Xt, e)
	});
	var Fn = /([\+\-]|\d\d)/gi;
	e.updateOffset = function() {};
	var Yn = /^(\-)?(?:(\d*)[. ])?(\d+)\:(\d+)(?:\:(\d+)(\.\d*)?)?$/,
	Rn = /^(-)?P(?:(-?[0-9,.]*)Y)?(?:(-?[0-9,.]*)M)?(?:(-?[0-9,.]*)W)?(?:(-?[0-9,.]*)D)?(?:T(?:(-?[0-9,.]*)H)?(?:(-?[0-9,.]*)M)?(?:(-?[0-9,.]*)S)?)?$/;
	Ge.fn = ze.prototype,
	Ge.invalid = function() {
		return Ge(NaN)
	};
	var Un = Xe(1, "add"),
	Bn = Xe( - 1, "subtract");
	e.defaultFormat = "YYYY-MM-DDTHH:mm:ssZ",
	e.defaultFormatUtc = "YYYY-MM-DDTHH:mm:ss[Z]";
	var Gn = x("moment().lang() is deprecated. Instead, use moment().localeData() to get the language configuration. Use moment().locale() to change languages.",
	function(e) {
		return void 0 === e ? this.localeData() : this.locale(e)
	});
	j(0, ["gg", 2], 0,
	function() {
		return this.weekYear() % 100
	}),
	j(0, ["GG", 2], 0,
	function() {
		return this.isoWeekYear() % 100
	}),
	nt("gggg", "weekYear"),
	nt("ggggg", "weekYear"),
	nt("GGGG", "isoWeekYear"),
	nt("GGGGG", "isoWeekYear"),
	T("weekYear", "gg"),
	T("isoWeekYear", "GG"),
	O("weekYear", 1),
	O("isoWeekYear", 1),
	H("G", Kt),
	H("g", Kt),
	H("GG", Ft, It),
	H("gg", Ft, It),
	H("GGGG", Bt, Wt),
	H("gggg", Bt, Wt),
	H("GGGGG", Gt, Ht),
	H("ggggg", Gt, Ht),
	B(["gggg", "ggggg", "GGGG", "GGGGG"],
	function(e, t, n, i) {
		t[i.substr(0, 2)] = b(e)
	}),
	B(["gg", "GG"],
	function(t, n, i, r) {
		n[r] = e.parseTwoDigitYear(t)
	}),
	j("Q", 0, "Qo", "quarter"),
	T("quarter", "Q"),
	O("quarter", 7),
	H("Q", Pt),
	U("Q",
	function(e, t) {
		t[rn] = 3 * (b(e) - 1)
	}),
	j("D", ["DD", 2], "Do", "date"),
	T("date", "D"),
	O("date", 9),
	H("D", Ft),
	H("DD", Ft, It),
	H("Do",
	function(e, t) {
		return e ? t._dayOfMonthOrdinalParse || t._ordinalParse: t._dayOfMonthOrdinalParseLenient
	}),
	U(["D", "DD"], on),
	U("Do",
	function(e, t) {
		t[on] = b(e.match(Ft)[0], 10)
	});
	var Vn = A("Date", !0);
	j("DDD", ["DDDD", 3], "DDDo", "dayOfYear"),
	T("dayOfYear", "DDD"),
	O("dayOfYear", 4),
	H("DDD", Ut),
	H("DDDD", zt),
	U(["DDD", "DDDD"],
	function(e, t, n) {
		n._dayOfYear = b(e)
	}),
	j("m", ["mm", 2], 0, "minute"),
	T("minute", "m"),
	O("minute", 14),
	H("m", Ft),
	H("mm", Ft, It),
	U(["m", "mm"], sn);
	var Kn = A("Minutes", !1);
	j("s", ["ss", 2], 0, "second"),
	T("second", "s"),
	O("second", 15),
	H("s", Ft),
	H("ss", Ft, It),
	U(["s", "ss"], ln);
	var Zn = A("Seconds", !1);
	j("S", 0, 0,
	function() {
		return~~ (this.millisecond() / 100)
	}),
	j(0, ["SS", 2], 0,
	function() {
		return~~ (this.millisecond() / 10)
	}),
	j(0, ["SSS", 3], 0, "millisecond"),
	j(0, ["SSSS", 4], 0,
	function() {
		return 10 * this.millisecond()
	}),
	j(0, ["SSSSS", 5], 0,
	function() {
		return 100 * this.millisecond()
	}),
	j(0, ["SSSSSS", 6], 0,
	function() {
		return 1e3 * this.millisecond()
	}),
	j(0, ["SSSSSSS", 7], 0,
	function() {
		return 1e4 * this.millisecond()
	}),
	j(0, ["SSSSSSSS", 8], 0,
	function() {
		return 1e5 * this.millisecond()
	}),
	j(0, ["SSSSSSSSS", 9], 0,
	function() {
		return 1e6 * this.millisecond()
	}),
	T("millisecond", "ms"),
	O("millisecond", 16),
	H("S", Ut, Pt),
	H("SS", Ut, It),
	H("SSS", Ut, zt);
	var Xn;
	for (Xn = "SSSS"; Xn.length <= 9; Xn += "S") H(Xn, Vt);
	for (Xn = "S"; Xn.length <= 9; Xn += "S") U(Xn,
	function(e, t) {
		t[cn] = b(1e3 * ("0." + e))
	});
	var Qn = A("Milliseconds", !1);
	j("z", 0, 0, "zoneAbbr"),
	j("zz", 0, 0, "zoneName");
	var Jn = g.prototype;
	Jn.add = Un,
	Jn.calendar = function(t, n) {
		var i = t || je(),
		r = Re(i, this).startOf("day"),
		o = e.calendarFormat(this, r) || "sameElse",
		a = n && (C(n[o]) ? n[o].call(this, i) : n[o]);
		return this.format(a || this.localeData().calendar(o, this, je(i)))
	},
	Jn.clone = function() {
		return new g(this)
	},
	Jn.diff = function(e, t, n) {
		var i, r, o, a;
		return this.isValid() && (i = Re(e, this)).isValid() ? (r = 6e4 * (i.utcOffset() - this.utcOffset()), "year" === (t = D(t)) || "month" === t || "quarter" === t ? (a = Je(this, i), "quarter" === t ? a /= 3 : "year" === t && (a /= 12)) : (o = this - i, a = "second" === t ? o / 1e3: "minute" === t ? o / 6e4: "hour" === t ? o / 36e5: "day" === t ? (o - r) / 864e5: "week" === t ? (o - r) / 6048e5: o), n ? a: y(a)) : NaN
	},
	Jn.endOf = function(e) {
		return void 0 === (e = D(e)) || "millisecond" === e ? this: ("date" === e && (e = "day"), this.startOf(e).add(1, "isoWeek" === e ? "week": e).subtract(1, "ms"))
	},
	Jn.format = function(t) {
		t || (t = this.isUtc() ? e.defaultFormatUtc: e.defaultFormat);
		var n = z(this, t);
		return this.localeData().postformat(n)
	},
	Jn.from = function(e, t) {
		return this.isValid() && (v(e) && e.isValid() || je(e).isValid()) ? Ge({
			to: this,
			from: e
		}).locale(this.locale()).humanize(!t) : this.localeData().invalidDate()
	},
	Jn.fromNow = function(e) {
		return this.from(je(), e)
	},
	Jn.to = function(e, t) {
		return this.isValid() && (v(e) && e.isValid() || je(e).isValid()) ? Ge({
			from: this,
			to: e
		}).locale(this.locale()).humanize(!t) : this.localeData().invalidDate()
	},
	Jn.toNow = function(e) {
		return this.to(je(), e)
	},
	Jn.get = function(e) {
		return e = D(e),
		C(this[e]) ? this[e]() : this
	},
	Jn.invalidAt = function() {
		return h(this).overflow
	},
	Jn.isAfter = function(e, t) {
		var n = v(e) ? e: je(e);
		return ! (!this.isValid() || !n.isValid()) && ("millisecond" === (t = D(r(t) ? "millisecond": t)) ? this.valueOf() > n.valueOf() : n.valueOf() < this.clone().startOf(t).valueOf())
	},
	Jn.isBefore = function(e, t) {
		var n = v(e) ? e: je(e);
		return ! (!this.isValid() || !n.isValid()) && ("millisecond" === (t = D(r(t) ? "millisecond": t)) ? this.valueOf() < n.valueOf() : this.clone().endOf(t).valueOf() < n.valueOf())
	},
	Jn.isBetween = function(e, t, n, i) {
		return ("(" === (i = i || "()")[0] ? this.isAfter(e, n) : !this.isBefore(e, n)) && (")" === i[1] ? this.isBefore(t, n) : !this.isAfter(t, n))
	},
	Jn.isSame = function(e, t) {
		var n, i = v(e) ? e: je(e);
		return ! (!this.isValid() || !i.isValid()) && ("millisecond" === (t = D(t || "millisecond")) ? this.valueOf() === i.valueOf() : (n = i.valueOf(), this.clone().startOf(t).valueOf() <= n && n <= this.clone().endOf(t).valueOf()))
	},
	Jn.isSameOrAfter = function(e, t) {
		return this.isSame(e, t) || this.isAfter(e, t)
	},
	Jn.isSameOrBefore = function(e, t) {
		return this.isSame(e, t) || this.isBefore(e, t)
	},
	Jn.isValid = function() {
		return f(this)
	},
	Jn.lang = Gn,
	Jn.locale = et,
	Jn.localeData = tt,
	Jn.max = Wn,
	Jn.min = zn,
	Jn.parsingFlags = function() {
		return c({},
		h(this))
	},
	Jn.set = function(e, t) {
		if ("object" == typeof e) for (var n = N(e = L(e)), i = 0; i < n.length; i++) this[n[i].unit](e[n[i].unit]);
		else if (e = D(e), C(this[e])) return this[e](t);
		return this
	},
	Jn.startOf = function(e) {
		switch (e = D(e)) {
		case "year":
			this.month(0);
		case "quarter":
		case "month":
			this.date(1);
		case "week":
		case "isoWeek":
		case "day":
		case "date":
			this.hours(0);
		case "hour":
			this.minutes(0);
		case "minute":
			this.seconds(0);
		case "second":
			this.milliseconds(0)
		}
		return "week" === e && this.weekday(0),
		"isoWeek" === e && this.isoWeekday(1),
		"quarter" === e && this.month(3 * Math.floor(this.month() / 3)),
		this
	},
	Jn.subtract = Bn,
	Jn.toArray = function() {
		var e = this;
		return [e.year(), e.month(), e.date(), e.hour(), e.minute(), e.second(), e.millisecond()]
	},
	Jn.toObject = function() {
		var e = this;
		return {
			years: e.year(),
			months: e.month(),
			date: e.date(),
			hours: e.hours(),
			minutes: e.minutes(),
			seconds: e.seconds(),
			milliseconds: e.milliseconds()
		}
	},
	Jn.toDate = function() {
		return new Date(this.valueOf())
	},
	Jn.toISOString = function() {
		if (!this.isValid()) return null;
		var e = this.clone().utc();
		return e.year() < 0 || e.year() > 9999 ? z(e, "YYYYYY-MM-DD[T]HH:mm:ss.SSS[Z]") : C(Date.prototype.toISOString) ? this.toDate().toISOString() : z(e, "YYYY-MM-DD[T]HH:mm:ss.SSS[Z]")
	},
	Jn.inspect = function() {
		if (!this.isValid()) return "moment.invalid(/* " + this._i + " */)";
		var e = "moment",
		t = "";
		this.isLocal() || (e = 0 === this.utcOffset() ? "moment.utc": "moment.parseZone", t = "Z");
		var n = "[" + e + '("]',
		i = 0 <= this.year() && this.year() <= 9999 ? "YYYY": "YYYYYY",
		r = t + '[")]';
		return this.format(n + i + "-MM-DD[T]HH:mm:ss.SSS" + r)
	},
	Jn.toJSON = function() {
		return this.isValid() ? this.toISOString() : null
	},
	Jn.toString = function() {
		return this.clone().locale("en").format("ddd MMM DD YYYY HH:mm:ss [GMT]ZZ")
	},
	Jn.unix = function() {
		return Math.floor(this.valueOf() / 1e3)
	},
	Jn.valueOf = function() {
		return this._d.valueOf() - 6e4 * (this._offset || 0)
	},
	Jn.creationData = function() {
		return {
			input: this._i,
			format: this._f,
			locale: this._locale,
			isUTC: this._isUTC,
			strict: this._strict
		}
	},
	Jn.year = yn,
	Jn.isLeapYear = function() {
		return ee(this.year())
	},
	Jn.weekYear = function(e) {
		return it.call(this, e, this.week(), this.weekday(), this.localeData()._week.dow, this.localeData()._week.doy)
	},
	Jn.isoWeekYear = function(e) {
		return it.call(this, e, this.isoWeek(), this.isoWeekday(), 1, 4)
	},
	Jn.quarter = Jn.quarters = function(e) {
		return null == e ? Math.ceil((this.month() + 1) / 3) : this.month(3 * (e - 1) + this.month() % 3)
	},
	Jn.month = X,
	Jn.daysInMonth = function() {
		return V(this.year(), this.month())
	},
	Jn.week = Jn.weeks = function(e) {
		var t = this.localeData().week(this);
		return null == e ? t: this.add(7 * (e - t), "d")
	},
	Jn.isoWeek = Jn.isoWeeks = function(e) {
		var t = oe(this, 1, 4).week;
		return null == e ? t: this.add(7 * (e - t), "d")
	},
	Jn.weeksInYear = function() {
		var e = this.localeData()._week;
		return ae(this.year(), e.dow, e.doy)
	},
	Jn.isoWeeksInYear = function() {
		return ae(this.year(), 1, 4)
	},
	Jn.date = Vn,
	Jn.day = Jn.days = function(e) {
		if (!this.isValid()) return null != e ? this: NaN;
		var t = this._isUTC ? this._d.getUTCDay() : this._d.getDay();
		return null != e ? (e = se(e, this.localeData()), this.add(e - t, "d")) : t
	},
	Jn.weekday = function(e) {
		if (!this.isValid()) return null != e ? this: NaN;
		var t = (this.day() + 7 - this.localeData()._week.dow) % 7;
		return null == e ? t: this.add(e - t, "d")
	},
	Jn.isoWeekday = function(e) {
		if (!this.isValid()) return null != e ? this: NaN;
		if (null != e) {
			var t = le(e, this.localeData());
			return this.day(this.day() % 7 ? t: t - 7)
		}
		return this.day() || 7
	},
	Jn.dayOfYear = function(e) {
		var t = Math.round((this.clone().startOf("day") - this.clone().startOf("year")) / 864e5) + 1;
		return null == e ? t: this.add(e - t, "d")
	},
	Jn.hour = Jn.hours = Dn,
	Jn.minute = Jn.minutes = Kn,
	Jn.second = Jn.seconds = Zn,
	Jn.millisecond = Jn.milliseconds = Qn,
	Jn.utcOffset = function(t, n, i) {
		var r, o = this._offset || 0;
		if (!this.isValid()) return null != t ? this: NaN;
		if (null != t) {
			if ("string" == typeof t) {
				if (null === (t = Ye(Xt, t))) return this
			} else Math.abs(t) < 16 && !i && (t *= 60);
			return ! this._isUTC && n && (r = Ue(this)),
			this._offset = t,
			this._isUTC = !0,
			null != r && this.add(r, "m"),
			o !== t && (!n || this._changeInProgress ? Qe(this, Ge(t - o, "m"), 1, !1) : this._changeInProgress || (this._changeInProgress = !0, e.updateOffset(this, !0), this._changeInProgress = null)),
			this
		}
		return this._isUTC ? o: Ue(this)
	},
	Jn.utc = function(e) {
		return this.utcOffset(0, e)
	},
	Jn.local = function(e) {
		return this._isUTC && (this.utcOffset(0, e), this._isUTC = !1, e && this.subtract(Ue(this), "m")),
		this
	},
	Jn.parseZone = function() {
		if (null != this._tzm) this.utcOffset(this._tzm, !1, !0);
		else if ("string" == typeof this._i) {
			var e = Ye(Zt, this._i);
			null != e ? this.utcOffset(e) : this.utcOffset(0, !0)
		}
		return this
	},
	Jn.hasAlignedHourOffset = function(e) {
		return !! this.isValid() && (e = e ? je(e).utcOffset() : 0, (this.utcOffset() - e) % 60 == 0)
	},
	Jn.isDST = function() {
		return this.utcOffset() > this.clone().month(0).utcOffset() || this.utcOffset() > this.clone().month(5).utcOffset()
	},
	Jn.isLocal = function() {
		return !! this.isValid() && !this._isUTC
	},
	Jn.isUtcOffset = function() {
		return !! this.isValid() && this._isUTC
	},
	Jn.isUtc = Be,
	Jn.isUTC = Be,
	Jn.zoneAbbr = function() {
		return this._isUTC ? "UTC": ""
	},
	Jn.zoneName = function() {
		return this._isUTC ? "Coordinated Universal Time": ""
	},
	Jn.dates = x("dates accessor is deprecated. Use date instead.", Vn),
	Jn.months = x("months accessor is deprecated. Use month instead", X),
	Jn.years = x("years accessor is deprecated. Use year instead", yn),
	Jn.zone = x("moment().zone is deprecated, use moment().utcOffset instead. http://momentjs.com/guides/#/warnings/zone/",
	function(e, t) {
		return null != e ? ("string" != typeof e && (e = -e), this.utcOffset(e, t), this) : -this.utcOffset()
	}),
	Jn.isDSTShifted = x("isDSTShifted is deprecated. See http://momentjs.com/guides/#/warnings/dst-shifted/ for more information",
	function() {
		if (!r(this._isDSTShifted)) return this._isDSTShifted;
		var e = {};
		if (m(e, this), (e = Ee(e))._a) {
			var t = e._isUTC ? u(e._a) : je(e._a);
			this._isDSTShifted = this.isValid() && w(e._a, t.toArray()) > 0
		} else this._isDSTShifted = !1;
		return this._isDSTShifted
	});
	var ei = M.prototype;
	ei.calendar = function(e, t, n) {
		var i = this._calendar[e] || this._calendar.sameElse;
		return C(i) ? i.call(t, n) : i
	},
	ei.longDateFormat = function(e) {
		var t = this._longDateFormat[e],
		n = this._longDateFormat[e.toUpperCase()];
		return t || !n ? t: (this._longDateFormat[e] = n.replace(/MMMM|MM|DD|dddd/g,
		function(e) {
			return e.slice(1)
		}), this._longDateFormat[e])
	},
	ei.invalidDate = function() {
		return this._invalidDate
	},
	ei.ordinal = function(e) {
		return this._ordinal.replace("%d", e)
	},
	ei.preparse = ot,
	ei.postformat = ot,
	ei.relativeTime = function(e, t, n, i) {
		var r = this._relativeTime[n];
		return C(r) ? r(e, t, n, i) : r.replace(/%d/i, e)
	},
	ei.pastFuture = function(e, t) {
		var n = this._relativeTime[e > 0 ? "future": "past"];
		return C(n) ? n(t) : n.replace(/%s/i, t)
	},
	ei.set = function(e) {
		var t, n;
		for (n in e) t = e[n],
		C(t) ? this[n] = t: this["_" + n] = t;
		this._config = e,
		this._dayOfMonthOrdinalParseLenient = new RegExp((this._dayOfMonthOrdinalParse.source || this._ordinalParse.source) + "|" + /\d{1,2}/.source)
	},
	ei.months = function(e, n) {
		return e ? t(this._months) ? this._months[e.month()] : this._months[(this._months.isFormat || fn).test(n) ? "format": "standalone"][e.month()] : t(this._months) ? this._months: this._months.standalone
	},
	ei.monthsShort = function(e, n) {
		return e ? t(this._monthsShort) ? this._monthsShort[e.month()] : this._monthsShort[fn.test(n) ? "format": "standalone"][e.month()] : t(this._monthsShort) ? this._monthsShort: this._monthsShort.standalone
	},
	ei.monthsParse = function(e, t, n) {
		var i, r, o;
		if (this._monthsParseExact) return K.call(this, e, t, n);
		for (this._monthsParse || (this._monthsParse = [], this._longMonthsParse = [], this._shortMonthsParse = []), i = 0; i < 12; i++) {
			if (r = u([2e3, i]), n && !this._longMonthsParse[i] && (this._longMonthsParse[i] = new RegExp("^" + this.months(r, "").replace(".", "") + "$", "i"), this._shortMonthsParse[i] = new RegExp("^" + this.monthsShort(r, "").replace(".", "") + "$", "i")), n || this._monthsParse[i] || (o = "^" + this.months(r, "") + "|^" + this.monthsShort(r, ""), this._monthsParse[i] = new RegExp(o.replace(".", ""), "i")), n && "MMMM" === t && this._longMonthsParse[i].test(e)) return i;
			if (n && "MMM" === t && this._shortMonthsParse[i].test(e)) return i;
			if (!n && this._monthsParse[i].test(e)) return i
		}
	},
	ei.monthsRegex = function(e) {
		return this._monthsParseExact ? (l(this, "_monthsRegex") || Q.call(this), e ? this._monthsStrictRegex: this._monthsRegex) : (l(this, "_monthsRegex") || (this._monthsRegex = vn), this._monthsStrictRegex && e ? this._monthsStrictRegex: this._monthsRegex)
	},
	ei.monthsShortRegex = function(e) {
		return this._monthsParseExact ? (l(this, "_monthsRegex") || Q.call(this), e ? this._monthsShortStrictRegex: this._monthsShortRegex) : (l(this, "_monthsShortRegex") || (this._monthsShortRegex = gn), this._monthsShortStrictRegex && e ? this._monthsShortStrictRegex: this._monthsShortRegex)
	},
	ei.week = function(e) {
		return oe(e, this._week.dow, this._week.doy).week
	},
	ei.firstDayOfYear = function() {
		return this._week.doy
	},
	ei.firstDayOfWeek = function() {
		return this._week.dow
	},
	ei.weekdays = function(e, n) {
		return e ? t(this._weekdays) ? this._weekdays[e.day()] : this._weekdays[this._weekdays.isFormat.test(n) ? "format": "standalone"][e.day()] : t(this._weekdays) ? this._weekdays: this._weekdays.standalone
	},
	ei.weekdaysMin = function(e) {
		return e ? this._weekdaysMin[e.day()] : this._weekdaysMin
	},
	ei.weekdaysShort = function(e) {
		return e ? this._weekdaysShort[e.day()] : this._weekdaysShort
	},
	ei.weekdaysParse = function(e, t, n) {
		var i, r, o;
		if (this._weekdaysParseExact) return ce.call(this, e, t, n);
		for (this._weekdaysParse || (this._weekdaysParse = [], this._minWeekdaysParse = [], this._shortWeekdaysParse = [], this._fullWeekdaysParse = []), i = 0; i < 7; i++) {
			if (r = u([2e3, 1]).day(i), n && !this._fullWeekdaysParse[i] && (this._fullWeekdaysParse[i] = new RegExp("^" + this.weekdays(r, "").replace(".", ".?") + "$", "i"), this._shortWeekdaysParse[i] = new RegExp("^" + this.weekdaysShort(r, "").replace(".", ".?") + "$", "i"), this._minWeekdaysParse[i] = new RegExp("^" + this.weekdaysMin(r, "").replace(".", ".?") + "$", "i")), this._weekdaysParse[i] || (o = "^" + this.weekdays(r, "") + "|^" + this.weekdaysShort(r, "") + "|^" + this.weekdaysMin(r, ""), this._weekdaysParse[i] = new RegExp(o.replace(".", ""), "i")), n && "dddd" === t && this._fullWeekdaysParse[i].test(e)) return i;
			if (n && "ddd" === t && this._shortWeekdaysParse[i].test(e)) return i;
			if (n && "dd" === t && this._minWeekdaysParse[i].test(e)) return i;
			if (!n && this._weekdaysParse[i].test(e)) return i
		}
	},
	ei.weekdaysRegex = function(e) {
		return this._weekdaysParseExact ? (l(this, "_weekdaysRegex") || ue.call(this), e ? this._weekdaysStrictRegex: this._weekdaysRegex) : (l(this, "_weekdaysRegex") || (this._weekdaysRegex = _n), this._weekdaysStrictRegex && e ? this._weekdaysStrictRegex: this._weekdaysRegex)
	},
	ei.weekdaysShortRegex = function(e) {
		return this._weekdaysParseExact ? (l(this, "_weekdaysRegex") || ue.call(this), e ? this._weekdaysShortStrictRegex: this._weekdaysShortRegex) : (l(this, "_weekdaysShortRegex") || (this._weekdaysShortRegex = Cn), this._weekdaysShortStrictRegex && e ? this._weekdaysShortStrictRegex: this._weekdaysShortRegex)
	},
	ei.weekdaysMinRegex = function(e) {
		return this._weekdaysParseExact ? (l(this, "_weekdaysRegex") || ue.call(this), e ? this._weekdaysMinStrictRegex: this._weekdaysMinRegex) : (l(this, "_weekdaysMinRegex") || (this._weekdaysMinRegex = Sn), this._weekdaysMinStrictRegex && e ? this._weekdaysMinStrictRegex: this._weekdaysMinRegex)
	},
	ei.isPM = function(e) {
		return "p" === (e + "").toLowerCase().charAt(0)
	},
	ei.meridiem = function(e, t, n) {
		return e > 11 ? n ? "pm": "PM": n ? "am": "AM"
	},
	ve("en", {
		dayOfMonthOrdinalParse: /\d{1,2}(th|st|nd|rd)/,
		ordinal: function(e) {
			var t = e % 10;
			return e + (1 === b(e % 100 / 10) ? "th": 1 === t ? "st": 2 === t ? "nd": 3 === t ? "rd": "th")
		}
	}),
	e.lang = x("moment.lang is deprecated. Use moment.locale instead.", ve),
	e.langData = x("moment.langData is deprecated. Use moment.localeData instead.", be);
	var ti = Math.abs,
	ni = ft("ms"),
	ii = ft("s"),
	ri = ft("m"),
	oi = ft("h"),
	ai = ft("d"),
	si = ft("w"),
	li = ft("M"),
	ci = ft("y"),
	ui = pt("milliseconds"),
	di = pt("seconds"),
	hi = pt("minutes"),
	fi = pt("hours"),
	pi = pt("days"),
	mi = pt("months"),
	gi = pt("years"),
	vi = Math.round,
	yi = {
		ss: 44,
		s: 45,
		m: 45,
		h: 22,
		d: 26,
		M: 11
	},
	bi = Math.abs,
	wi = ze.prototype;
	return wi.isValid = function() {
		return this._isValid
	},
	wi.abs = function() {
		var e = this._data;
		return this._milliseconds = ti(this._milliseconds),
		this._days = ti(this._days),
		this._months = ti(this._months),
		e.milliseconds = ti(e.milliseconds),
		e.seconds = ti(e.seconds),
		e.minutes = ti(e.minutes),
		e.hours = ti(e.hours),
		e.months = ti(e.months),
		e.years = ti(e.years),
		this
	},
	wi.add = function(e, t) {
		return ct(this, e, t, 1)
	},
	wi.subtract = function(e, t) {
		return ct(this, e, t, -1)
	},
	wi.as = function(e) {
		if (!this.isValid()) return NaN;
		var t, n, i = this._milliseconds;
		if ("month" === (e = D(e)) || "year" === e) return t = this._days + i / 864e5,
		n = this._months + dt(t),
		"month" === e ? n: n / 12;
		switch (t = this._days + Math.round(ht(this._months)), e) {
		case "week":
			return t / 7 + i / 6048e5;
		case "day":
			return t + i / 864e5;
		case "hour":
			return 24 * t + i / 36e5;
		case "minute":
			return 1440 * t + i / 6e4;
		case "second":
			return 86400 * t + i / 1e3;
		case "millisecond":
			return Math.floor(864e5 * t) + i;
		default:
			throw new Error("Unknown unit " + e)
		}
	},
	wi.asMilliseconds = ni,
	wi.asSeconds = ii,
	wi.asMinutes = ri,
	wi.asHours = oi,
	wi.asDays = ai,
	wi.asWeeks = si,
	wi.asMonths = li,
	wi.asYears = ci,
	wi.valueOf = function() {
		return this.isValid() ? this._milliseconds + 864e5 * this._days + this._months % 12 * 2592e6 + 31536e6 * b(this._months / 12) : NaN
	},
	wi._bubble = function() {
		var e, t, n, i, r, o = this._milliseconds,
		a = this._days,
		s = this._months,
		l = this._data;
		return o >= 0 && a >= 0 && s >= 0 || o <= 0 && a <= 0 && s <= 0 || (o += 864e5 * ut(ht(s) + a), a = 0, s = 0),
		l.milliseconds = o % 1e3,
		e = y(o / 1e3),
		l.seconds = e % 60,
		t = y(e / 60),
		l.minutes = t % 60,
		n = y(t / 60),
		l.hours = n % 24,
		a += y(n / 24),
		r = y(dt(a)),
		s += r,
		a -= ut(ht(r)),
		i = y(s / 12),
		s %= 12,
		l.days = a,
		l.months = s,
		l.years = i,
		this
	},
	wi.get = function(e) {
		return e = D(e),
		this.isValid() ? this[e + "s"]() : NaN
	},
	wi.milliseconds = ui,
	wi.seconds = di,
	wi.minutes = hi,
	wi.hours = fi,
	wi.days = pi,
	wi.weeks = function() {
		return y(this.days() / 7)
	},
	wi.months = mi,
	wi.years = gi,
	wi.humanize = function(e) {
		if (!this.isValid()) return this.localeData().invalidDate();
		var t = this.localeData(),
		n = gt(this, !e, t);
		return e && (n = t.pastFuture( + this, n)),
		t.postformat(n)
	},
	wi.toISOString = vt,
	wi.toString = vt,
	wi.toJSON = vt,
	wi.locale = et,
	wi.localeData = tt,
	wi.toIsoString = x("toIsoString() is deprecated. Please use toISOString() instead (notice the capitals)", vt),
	wi.lang = Gn,
	j("X", 0, 0, "unix"),
	j("x", 0, 0, "valueOf"),
	H("x", Kt),
	H("X", Qt),
	U("X",
	function(e, t, n) {
		n._d = new Date(1e3 * parseFloat(e, 10))
	}),
	U("x",
	function(e, t, n) {
		n._d = new Date(b(e))
	}),
	e.version = "2.18.1",
	function(e) {
		yt = e
	} (je),
	e.fn = Jn,
	e.min = function() {
		return Pe("isBefore", [].slice.call(arguments, 0))
	},
	e.max = function() {
		return Pe("isAfter", [].slice.call(arguments, 0))
	},
	e.now = function() {
		return Date.now ? Date.now() : +new Date
	},
	e.utc = u,
	e.unix = function(e) {
		return je(1e3 * e)
	},
	e.months = function(e, t) {
		return st(e, t, "months")
	},
	e.isDate = a,
	e.locale = ve,
	e.invalid = p,
	e.duration = Ge,
	e.isMoment = v,
	e.weekdays = function(e, t, n) {
		return lt(e, t, n, "weekdays")
	},
	e.parseZone = function() {
		return je.apply(null, arguments).parseZone()
	},
	e.localeData = be,
	e.isDuration = We,
	e.monthsShort = function(e, t) {
		return st(e, t, "monthsShort")
	},
	e.weekdaysMin = function(e, t, n) {
		return lt(e, t, n, "weekdaysMin")
	},
	e.defineLocale = ye,
	e.updateLocale = function(e, t) {
		if (null != t) {
			var n, i = Ln;
			null != On[e] && (i = On[e]._config),
			(n = new M(t = S(i, t))).parentLocale = On[e],
			On[e] = n,
			ve(e)
		} else null != On[e] && (null != On[e].parentLocale ? On[e] = On[e].parentLocale: null != On[e] && delete On[e]);
		return On[e]
	},
	e.locales = function() {
		return Mt(On)
	},
	e.weekdaysShort = function(e, t, n) {
		return lt(e, t, n, "weekdaysShort")
	},
	e.normalizeUnits = D,
	e.relativeTimeRounding = function(e) {
		return void 0 === e ? vi: "function" == typeof e && (vi = e, !0)
	},
	e.relativeTimeThreshold = function(e, t) {
		return void 0 !== yi[e] && (void 0 === t ? yi[e] : (yi[e] = t, "s" === e && (yi.ss = t - 1), !0))
	},
	e.calendarFormat = function(e, t) {
		var n = e.diff(t, "days", !0);
		return n < -6 ? "sameElse": n < -1 ? "lastWeek": n < 0 ? "lastDay": n < 1 ? "sameDay": n < 2 ? "nextDay": n < 7 ? "nextWeek": "sameElse"
	},
	e.prototype = Jn,
	e
}),
function() {
	function e(e) {
		this.tokens = [],
		this.tokens.links = {},
		this.options = e || c.defaults,
		this.rules = u.normal,
		this.options.gfm && (this.options.tables ? this.rules = u.tables: this.rules = u.gfm)
	}
	function t(e, t) {
		if (this.options = t || c.defaults, this.links = e, this.rules = d.normal, this.renderer = this.options.renderer || new n, this.renderer.options = this.options, !this.links) throw new Error("Tokens array requires a `links` property.");
		this.options.gfm ? this.options.breaks ? this.rules = d.breaks: this.rules = d.gfm: this.options.pedantic && (this.rules = d.pedantic)
	}
	function n(e) {
		this.options = e || {}
	}
	function i(e) {
		this.tokens = [],
		this.token = null,
		this.options = e || c.defaults,
		this.options.renderer = this.options.renderer || new n,
		this.renderer = this.options.renderer,
		this.renderer.options = this.options
	}
	function r(e, t) {
		return e.replace(t ? /&/g: /&(?!#?\w+;)/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#39;")
	}
	function o(e) {
		return e.replace(/&([#\w]+);/g,
		function(e, t) {
			return "colon" === (t = t.toLowerCase()) ? ":": "#" === t.charAt(0) ? "x" === t.charAt(1) ? String.fromCharCode(parseInt(t.substring(2), 16)) : String.fromCharCode( + t.substring(1)) : ""
		})
	}
	function a(e, t) {
		return e = e.source,
		t = t || "",
		function n(i, r) {
			return i ? (r = r.source || r, r = r.replace(/(^|[^\[])\^/g, "$1"), e = e.replace(i, r), n) : new RegExp(e, t)
		}
	}
	function s() {}
	function l(e) {
		for (var t, n, i = 1; i < arguments.length; i++) {
			t = arguments[i];
			for (n in t) Object.prototype.hasOwnProperty.call(t, n) && (e[n] = t[n])
		}
		return e
	}
	function c(t, n, o) {
		if (o || "function" == typeof n) {
			o || (o = n, n = null);
			var a, s, u = (n = l({},
			c.defaults, n || {})).highlight,
			d = 0;
			try {
				a = e.lex(t, n)
			} catch(e) {
				return o(e)
			}
			s = a.length;
			var h = function(e) {
				if (e) return n.highlight = u,
				o(e);
				var t;
				try {
					t = i.parse(a, n)
				} catch(t) {
					e = t
				}
				return n.highlight = u,
				e ? o(e) : o(null, t)
			};
			if (!u || u.length < 3) return h();
			if (delete n.highlight, !s) return h();
			for (; d < a.length; d++) !
			function(e) {
				"code" !== e.type ? --s || h() : u(e.text, e.lang,
				function(t, n) {
					return t ? h(t) : null == n || n === e.text ? --s || h() : (e.text = n, e.escaped = !0, void(--s || h()))
				})
			} (a[d])
		} else try {
			return n && (n = l({},
			c.defaults, n)),
			i.parse(e.lex(t, n), n)
		} catch(e) {
			if (e.message += "\nPlease report this to https://github.com/chjj/marked.", (n || c.defaults).silent) return "<p>An error occured:</p><pre>" + r(e.message + "", !0) + "</pre>";
			throw e
		}
	}
	var u = {
		newline: /^\n+/,
		code: /^( {4}[^\n]+\n*)+/,
		fences: s,
		hr: /^( *[-*_]){3,} *(?:\n+|$)/,
		heading: /^ *(#{1,6}) *([^\n]+?) *#* *(?:\n+|$)/,
		nptable: s,
		lheading: /^([^\n]+)\n *(=|-){2,} *(?:\n+|$)/,
		blockquote: /^( *>[^\n]+(\n(?!def)[^\n]+)*\n*)+/,
		list: /^( *)(bull) [\s\S]+?(?:hr|def|\n{2,}(?! )(?!\1bull )\n*|\s*$)/,
		html: /^ *(?:comment *(?:\n|\s*$)|closed *(?:\n{2,}|\s*$)|closing *(?:\n{2,}|\s*$))/,
		def: /^ *\[([^\]]+)\]: *<?([^\s>]+)>?(?: +["(]([^\n]+)[")])? *(?:\n+|$)/,
		table: s,
		paragraph: /^((?:[^\n]+\n?(?!hr|heading|lheading|blockquote|tag|def))+)\n*/,
		text: /^[^\n]+/
	};
	u.bullet = /(?:[*+-]|\d+\.)/,
	u.item = /^( *)(bull) [^\n]*(?:\n(?!\1bull )[^\n]*)*/,
	u.item = a(u.item, "gm")(/bull/g, u.bullet)(),
	u.list = a(u.list)(/bull/g, u.bullet)("hr", "\\n+(?=\\1?(?:[-*_] *){3,}(?:\\n+|$))")("def", "\\n+(?=" + u.def.source + ")")(),
	u.blockquote = a(u.blockquote)("def", u.def)(),
	u._tag = "(?!(?:a|em|strong|small|s|cite|q|dfn|abbr|data|time|code|var|samp|kbd|sub|sup|i|b|u|mark|ruby|rt|rp|bdi|bdo|span|br|wbr|ins|del|img)\\b)\\w+(?!:/|[^\\w\\s@]*@)\\b",
	u.html = a(u.html)("comment", /<!--[\s\S]*?-->/)("closed", /<(tag)[\s\S]+?<\/\1>/)("closing", /<tag(?:"[^"]*"|'[^']*'|[^'">])*?>/)(/tag/g, u._tag)(),
	u.paragraph = a(u.paragraph)("hr", u.hr)("heading", u.heading)("lheading", u.lheading)("blockquote", u.blockquote)("tag", "<" + u._tag)("def", u.def)(),
	u.normal = l({},
	u),
	u.gfm = l({},
	u.normal, {
		fences: /^ *(`{3,}|~{3,}) *(\S+)? *\n([\s\S]+?)\s*\1 *(?:\n+|$)/,
		paragraph: /^/
	}),
	u.gfm.paragraph = a(u.paragraph)("(?!", "(?!" + u.gfm.fences.source.replace("\\1", "\\2") + "|" + u.list.source.replace("\\1", "\\3") + "|")(),
	u.tables = l({},
	u.gfm, {
		nptable: /^ *(\S.*\|.*)\n *([-:]+ *\|[-| :]*)\n((?:.*\|.*(?:\n|$))*)\n*/,
		table: /^ *\|(.+)\n *\|( *[-:]+[-| :]*)\n((?: *\|.*(?:\n|$))*)\n*/
	}),
	e.rules = u,
	e.lex = function(t, n) {
		return new e(n).lex(t)
	},
	e.prototype.lex = function(e) {
		return e = e.replace(/\r\n|\r/g, "\n").replace(/\t/g, "    ").replace(/\u00a0/g, " ").replace(/\u2424/g, "\n"),
		this.token(e, !0)
	},
	e.prototype.token = function(e, t, n) {
		for (var i, r, o, a, s, l, c, d, h, e = e.replace(/^ +$/gm, ""); e;) if ((o = this.rules.newline.exec(e)) && (e = e.substring(o[0].length), o[0].length > 1 && this.tokens.push({
			type: "space"
		})), o = this.rules.code.exec(e)) e = e.substring(o[0].length),
		o = o[0].replace(/^ {4}/gm, ""),
		this.tokens.push({
			type: "code",
			text: this.options.pedantic ? o: o.replace(/\n+$/, "")
		});
		else if (o = this.rules.fences.exec(e)) e = e.substring(o[0].length),
		this.tokens.push({
			type: "code",
			lang: o[2],
			text: o[3]
		});
		else if (o = this.rules.heading.exec(e)) e = e.substring(o[0].length),
		this.tokens.push({
			type: "heading",
			depth: o[1].length,
			text: o[2]
		});
		else if (t && (o = this.rules.nptable.exec(e))) {
			for (e = e.substring(o[0].length), l = {
				type: "table",
				header: o[1].replace(/^ *| *\| *$/g, "").split(/ *\| */),
				align: o[2].replace(/^ *|\| *$/g, "").split(/ *\| */),
				cells: o[3].replace(/\n$/, "").split("\n")
			},
			d = 0; d < l.align.length; d++) / ^*-+:*$ / .test(l.align[d]) ? l.align[d] = "right": /^ *:-+: *$/.test(l.align[d]) ? l.align[d] = "center": /^ *:-+ *$/.test(l.align[d]) ? l.align[d] = "left": l.align[d] = null;
			for (d = 0; d < l.cells.length; d++) l.cells[d] = l.cells[d].split(/ *\| */);
			this.tokens.push(l)
		} else if (o = this.rules.lheading.exec(e)) e = e.substring(o[0].length),
		this.tokens.push({
			type: "heading",
			depth: "=" === o[2] ? 1 : 2,
			text: o[1]
		});
		else if (o = this.rules.hr.exec(e)) e = e.substring(o[0].length),
		this.tokens.push({
			type: "hr"
		});
		else if (o = this.rules.blockquote.exec(e)) e = e.substring(o[0].length),
		this.tokens.push({
			type: "blockquote_start"
		}),
		o = o[0].replace(/^ *> ?/gm, ""),
		this.token(o, t, !0),
		this.tokens.push({
			type: "blockquote_end"
		});
		else if (o = this.rules.list.exec(e)) {
			for (e = e.substring(o[0].length), a = o[2], this.tokens.push({
				type: "list_start",
				ordered: a.length > 1
			}), i = !1, h = (o = o[0].match(this.rules.item)).length, d = 0; d < h; d++) c = (l = o[d]).length,
			~ (l = l.replace(/^ *([*+-]|\d+\.) +/, "")).indexOf("\n ") && (c -= l.length, l = this.options.pedantic ? l.replace(/^ {1,4}/gm, "") : l.replace(new RegExp("^ {1," + c + "}", "gm"), "")),
			this.options.smartLists && d !== h - 1 && (a === (s = u.bullet.exec(o[d + 1])[0]) || a.length > 1 && s.length > 1 || (e = o.slice(d + 1).join("\n") + e, d = h - 1)),
			r = i || /\n\n(?!\s*$)/.test(l),
			d !== h - 1 && (i = "\n" === l.charAt(l.length - 1), r || (r = i)),
			this.tokens.push({
				type: r ? "loose_item_start": "list_item_start"
			}),
			this.token(l, !1, n),
			this.tokens.push({
				type: "list_item_end"
			});
			this.tokens.push({
				type: "list_end"
			})
		} else if (o = this.rules.html.exec(e)) e = e.substring(o[0].length),
		this.tokens.push({
			type: this.options.sanitize ? "paragraph": "html",
			pre: "pre" === o[1] || "script" === o[1] || "style" === o[1],
			text: o[0]
		});
		else if (!n && t && (o = this.rules.def.exec(e))) e = e.substring(o[0].length),
		this.tokens.links[o[1].toLowerCase()] = {
			href: o[2],
			title: o[3]
		};
		else if (t && (o = this.rules.table.exec(e))) {
			for (e = e.substring(o[0].length), l = {
				type: "table",
				header: o[1].replace(/^ *| *\| *$/g, "").split(/ *\| */),
				align: o[2].replace(/^ *|\| *$/g, "").split(/ *\| */),
				cells: o[3].replace(/(?: *\| *)?\n$/, "").split("\n")
			},
			d = 0; d < l.align.length; d++) / ^*-+:*$ / .test(l.align[d]) ? l.align[d] = "right": /^ *:-+: *$/.test(l.align[d]) ? l.align[d] = "center": /^ *:-+ *$/.test(l.align[d]) ? l.align[d] = "left": l.align[d] = null;
			for (d = 0; d < l.cells.length; d++) l.cells[d] = l.cells[d].replace(/^ *\| *| *\| *$/g, "").split(/ *\| */);
			this.tokens.push(l)
		} else if (t && (o = this.rules.paragraph.exec(e))) e = e.substring(o[0].length),
		this.tokens.push({
			type: "paragraph",
			text: "\n" === o[1].charAt(o[1].length - 1) ? o[1].slice(0, -1) : o[1]
		});
		else if (o = this.rules.text.exec(e)) e = e.substring(o[0].length),
		this.tokens.push({
			type: "text",
			text: o[0]
		});
		else if (e) throw new Error("Infinite loop on byte: " + e.charCodeAt(0));
		return this.tokens
	};
	var d = {
		escape: /^\\([\\`*{}\[\]()#+\-.!_>])/,
		autolink: /^<([^ >]+(@|:\/)[^ >]+)>/,
		url: s,
		tag: /^<!--[\s\S]*?-->|^<\/?\w+(?:"[^"]*"|'[^']*'|[^'">])*?>/,
		link: /^!?\[(inside)\]\(href\)/,
		reflink: /^!?\[(inside)\]\s*\[([^\]]*)\]/,
		nolink: /^!?\[((?:\[[^\]]*\]|[^\[\]])*)\]/,
		strong: /^__([\s\S]+?)__(?!_)|^\*\*([\s\S]+?)\*\*(?!\*)/,
		em: /^\b_((?:__|[\s\S])+?)_\b|^\*((?:\*\*|[\s\S])+?)\*(?!\*)/,
		code: /^(`+)\s*([\s\S]*?[^`])\s*\1(?!`)/,
		br: /^ {2,}\n(?!\s*$)/,
		del: s,
		text: /^[\s\S]+?(?=[\\<!\[_*`]| {2,}\n|$)/
	};
	d._inside = /(?:\[[^\]]*\]|[^\[\]]|\](?=[^\[]*\]))*/,
	d._href = /\s*<?([\s\S]*?)>?(?:\s+['"]([\s\S]*?)['"])?\s*/,
	d.link = a(d.link)("inside", d._inside)("href", d._href)(),
	d.reflink = a(d.reflink)("inside", d._inside)(),
	d.normal = l({},
	d),
	d.pedantic = l({},
	d.normal, {
		strong: /^__(?=\S)([\s\S]*?\S)__(?!_)|^\*\*(?=\S)([\s\S]*?\S)\*\*(?!\*)/,
		em: /^_(?=\S)([\s\S]*?\S)_(?!_)|^\*(?=\S)([\s\S]*?\S)\*(?!\*)/
	}),
	d.gfm = l({},
	d.normal, {
		escape: a(d.escape)("])", "~|])")(),
		url: /^(https?:\/\/[^\s<]+[^<.,:;"')\]\s])/,
		del: /^~~(?=\S)([\s\S]*?\S)~~/,
		text: a(d.text)("]|", "~]|")("|", "|https?://|")()
	}),
	d.breaks = l({},
	d.gfm, {
		br: a(d.br)("{2,}", "*")(),
		text: a(d.gfm.text)("{2,}", "*")()
	}),
	t.rules = d,
	t.output = function(e, n, i) {
		return new t(n, i).output(e)
	},
	t.prototype.output = function(e) {
		for (var t, n, i, o, a = ""; e;) if (o = this.rules.escape.exec(e)) e = e.substring(o[0].length),
		a += o[1];
		else if (o = this.rules.autolink.exec(e)) e = e.substring(o[0].length),
		"@" === o[2] ? (n = ":" === o[1].charAt(6) ? this.mangle(o[1].substring(7)) : this.mangle(o[1]), i = this.mangle("mailto:") + n) : i = n = r(o[1]),
		a += this.renderer.link(i, null, n);
		else if (this.inLink || !(o = this.rules.url.exec(e))) {
			if (o = this.rules.tag.exec(e)) ! this.inLink && /^<a /i.test(o[0]) ? this.inLink = !0 : this.inLink && /^<\/a>/i.test(o[0]) && (this.inLink = !1),
			e = e.substring(o[0].length),
			a += this.options.sanitize ? r(o[0]) : o[0];
			else if (o = this.rules.link.exec(e)) e = e.substring(o[0].length),
			this.inLink = !0,
			a += this.outputLink(o, {
				href: o[2],
				title: o[3]
			}),
			this.inLink = !1;
			else if ((o = this.rules.reflink.exec(e)) || (o = this.rules.nolink.exec(e))) {
				if (e = e.substring(o[0].length), t = (o[2] || o[1]).replace(/\s+/g, " "), !(t = this.links[t.toLowerCase()]) || !t.href) {
					a += o[0].charAt(0),
					e = o[0].substring(1) + e;
					continue
				}
				this.inLink = !0,
				a += this.outputLink(o, t),
				this.inLink = !1
			} else if (o = this.rules.strong.exec(e)) e = e.substring(o[0].length),
			a += this.renderer.strong(this.output(o[2] || o[1]));
			else if (o = this.rules.em.exec(e)) e = e.substring(o[0].length),
			a += this.renderer.em(this.output(o[2] || o[1]));
			else if (o = this.rules.code.exec(e)) e = e.substring(o[0].length),
			a += this.renderer.codespan(r(o[2], !0));
			else if (o = this.rules.br.exec(e)) e = e.substring(o[0].length),
			a += this.renderer.br();
			else if (o = this.rules.del.exec(e)) e = e.substring(o[0].length),
			a += this.renderer.del(this.output(o[1]));
			else if (o = this.rules.text.exec(e)) e = e.substring(o[0].length),
			a += r(this.smartypants(o[0]));
			else if (e) throw new Error("Infinite loop on byte: " + e.charCodeAt(0))
		} else e = e.substring(o[0].length),
		i = n = r(o[1]),
		a += this.renderer.link(i, null, n);
		return a
	},
	t.prototype.outputLink = function(e, t) {
		var n = r(t.href),
		i = t.title ? r(t.title) : null;
		return "!" !== e[0].charAt(0) ? this.renderer.link(n, i, this.output(e[1])) : this.renderer.image(n, i, r(e[1]))
	},
	t.prototype.smartypants = function(e) {
		return this.options.smartypants ? e.replace(/--/g, "—").replace(/(^|[-\u2014/ (\ [{\s])'/g,"$1‘").replace(/'/g,"’").replace(/(^|[-\u2014/(\[{\u2018\s]) /g,"$1“").replace(/"/g, "”").replace(/\.{3}/g, "…") : e
		},
		t.prototype.mangle = function(e) {
			for (var t, n = "",
			i = e.length,
			r = 0; r < i; r++) t = e.charCodeAt(r),
			Math.random() > .5 && (t = "x" + t.toString(16)),
			n += "&#" + t + ";";
			return n
		},
		n.prototype.code = function(e, t, n) {
			if (this.options.highlight) {
				var i = this.options.highlight(e, t);
				null != i && i !== e && (n = !0, e = i)
			}
			return t ? '<pre><code class="' + this.options.langPrefix + r(t, !0) + '">' + (n ? e: r(e, !0)) + "\n</code></pre>\n": "<pre><code>" + (n ? e: r(e, !0)) + "\n</code></pre>"
		},
		n.prototype.blockquote = function(e) {
			return "<blockquote>\n" + e + "</blockquote>\n"
		},
		n.prototype.html = function(e) {
			return e
		},
		n.prototype.heading = function(e, t, n) {
			return "<h" + t + ' id="' + this.options.headerPrefix + n.toLowerCase().replace(/[^\w]+/g, "-") + '">' + e + "</h" + t + ">\n"
		},
		n.prototype.hr = function() {
			return this.options.xhtml ? "<hr/>\n": "<hr>\n"
		},
		n.prototype.list = function(e, t) {
			var n = t ? "ol": "ul";
			return "<" + n + ">\n" + e + "</" + n + ">\n"
		},
		n.prototype.listitem = function(e) {
			return "<li>" + e + "</li>\n"
		},
		n.prototype.paragraph = function(e) {
			return "<p>" + e + "</p>\n"
		},
		n.prototype.table = function(e, t) {
			return "<table>\n<thead>\n" + e + "</thead>\n<tbody>\n" + t + "</tbody>\n</table>\n"
		},
		n.prototype.tablerow = function(e) {
			return "<tr>\n" + e + "</tr>\n"
		},
		n.prototype.tablecell = function(e, t) {
			var n = t.header ? "th": "td";
			return (t.align ? "<" + n + ' style="text-align:' + t.align + '">': "<" + n + ">") + e + "</" + n + ">\n"
		},
		n.prototype.strong = function(e) {
			return "<strong>" + e + "</strong>"
		},
		n.prototype.em = function(e) {
			return "<em>" + e + "</em>"
		},
		n.prototype.codespan = function(e) {
			return "<code>" + e + "</code>"
		},
		n.prototype.br = function() {
			return this.options.xhtml ? "<br/>": "<br>"
		},
		n.prototype.del = function(e) {
			return "<del>" + e + "</del>"
		},
		n.prototype.link = function(e, t, n) {
			if (this.options.sanitize) {
				try {
					var i = decodeURIComponent(o(e)).replace(/[^\w:]/g, "").toLowerCase()
				} catch(e) {
					return ""
				}
				if (0 === i.indexOf("javascript:")) return ""
			}
			var r = '<a href="' + e + '"';
			return t && (r += ' title="' + t + '"'),
			r += ">" + n + "</a>"
		},
		n.prototype.image = function(e, t, n) {
			var i = '<img src="' + e + '" alt="' + n + '"';
			return t && (i += ' title="' + t + '"'),
			i += this.options.xhtml ? "/>": ">"
		},
		i.parse = function(e, t, n) {
			return new i(t, n).parse(e)
		},
		i.prototype.parse = function(e) {
			this.inline = new t(e.links, this.options, this.renderer),
			this.tokens = e.reverse();
			for (var n = ""; this.next();) n += this.tok();
			return n
		},
		i.prototype.next = function() {
			return this.token = this.tokens.pop()
		},
		i.prototype.peek = function() {
			return this.tokens[this.tokens.length - 1] || 0
		},
		i.prototype.parseText = function() {
			for (var e = this.token.text;
			"text" === this.peek().type;) e += "\n" + this.next().text;
			return this.inline.output(e)
		},
		i.prototype.tok = function() {
			switch (this.token.type) {
			case "space":
				return "";
			case "hr":
				return this.renderer.hr();
			case "heading":
				return this.renderer.heading(this.inline.output(this.token.text), this.token.depth, this.token.text);
			case "code":
				return this.renderer.code(this.token.text, this.token.lang, this.token.escaped);
			case "table":
				var e, t, n, i, r = "",
				o = "";
				for (n = "", e = 0; e < this.token.header.length; e++)({
					header: !0,
					align: this.token.align[e]
				}),
				n += this.renderer.tablecell(this.inline.output(this.token.header[e]), {
					header: !0,
					align: this.token.align[e]
				});
				for (r += this.renderer.tablerow(n), e = 0; e < this.token.cells.length; e++) {
					for (t = this.token.cells[e], n = "", i = 0; i < t.length; i++) n += this.renderer.tablecell(this.inline.output(t[i]), {
						header: !1,
						align: this.token.align[i]
					});
					o += this.renderer.tablerow(n)
				}
				return this.renderer.table(r, o);
			case "blockquote_start":
				for (o = "";
				"blockquote_end" !== this.next().type;) o += this.tok();
				return this.renderer.blockquote(o);
			case "list_start":
				for (var o = "",
				a = this.token.ordered;
				"list_end" !== this.next().type;) o += this.tok();
				return this.renderer.list(o, a);
			case "list_item_start":
				for (o = "";
				"list_item_end" !== this.next().type;) o += "text" === this.token.type ? this.parseText() : this.tok();
				return this.renderer.listitem(o);
			case "loose_item_start":
				for (o = "";
				"list_item_end" !== this.next().type;) o += this.tok();
				return this.renderer.listitem(o);
			case "html":
				var s = this.token.pre || this.options.pedantic ? this.token.text: this.inline.output(this.token.text);
				return this.renderer.html(s);
			case "paragraph":
				return this.renderer.paragraph(this.inline.output(this.token.text));
			case "text":
				return this.renderer.paragraph(this.parseText())
			}
		},
		s.exec = s, c.options = c.setOptions = function(e) {
			return l(c.defaults, e),
			c
		},
		c.defaults = {
			gfm: !0,
			tables: !0,
			breaks: !1,
			pedantic: !1,
			sanitize: !1,
			smartLists: !1,
			silent: !1,
			highlight: null,
			langPrefix: "lang-",
			smartypants: !1,
			headerPrefix: "",
			renderer: new n,
			xhtml: !1
		},
		c.Parser = i, c.parser = i.parse, c.Renderer = n, c.Lexer = e, c.lexer = e.lex, c.InlineLexer = t, c.inlineLexer = t.output, c.parse = c, "undefined" != typeof module && "object" == typeof exports ? module.exports = c: "function" == typeof define && define.amd ? define(function() {
			return c
		}) : this.marked = c
	}.call(function() {
		return this || ("undefined" != typeof window ? window: global)
	} ()),
	function(e) {
		if ("function" == typeof define && define.amd && define("uikit",
		function() {
			var t = window.UIkit || e(window, window.jQuery, window.document);
			return t.load = function(e, n, i, r) {
				var o, a = e.split(","),
				s = [],
				l = (r.config && r.config.uikit && r.config.uikit.base ? r.config.uikit.base: "").replace(/\/+$/g, "");
				if (!l) throw new Error("Please define base path to UIkit in the requirejs config.");
				for (o = 0; o < a.length; o += 1) {
					var c = a[o].replace(/\./g, "/");
					s.push(l + "/components/" + c)
				}
				n(s,
				function() {
					i(t)
				})
			},
			t
		}), !window.jQuery) throw new Error("UIkit requires jQuery");
		window && window.jQuery && e(window, window.jQuery, window.document)
	} (function(e, t, n) {
		"use strict";
		var i = {},
		r = e.UIkit ? Object.create(e.UIkit) : void 0;
		if (i.version = "2.27.2", i.noConflict = function() {
			return r && (e.UIkit = r, t.UIkit = r, t.fn.uk = r.fn),
			i
		},
		i.prefix = function(e) {
			return e
		},
		i.$ = t, i.$doc = i.$(document), i.$win = i.$(window), i.$html = i.$("html"), i.support = {},
		i.support.transition = function() {
			var e = function() {
				var e, t = n.body || n.documentElement,
				i = {
					WebkitTransition: "webkitTransitionEnd",
					MozTransition: "transitionend",
					OTransition: "oTransitionEnd otransitionend",
					transition: "transitionend"
				};
				for (e in i) if (void 0 !== t.style[e]) return i[e]
			} ();
			return e && {
				end: e
			}
		} (), i.support.animation = function() {
			var e = function() {
				var e, t = n.body || n.documentElement,
				i = {
					WebkitAnimation: "webkitAnimationEnd",
					MozAnimation: "animationend",
					OAnimation: "oAnimationEnd oanimationend",
					animation: "animationend"
				};
				for (e in i) if (void 0 !== t.style[e]) return i[e]
			} ();
			return e && {
				end: e
			}
		} (),
		function() {
			Date.now = Date.now ||
			function() {
				return (new Date).getTime()
			};
			for (var e = ["webkit", "moz"], t = 0; t < e.length && !window.requestAnimationFrame; ++t) {
				var n = e[t];
				window.requestAnimationFrame = window[n + "RequestAnimationFrame"],
				window.cancelAnimationFrame = window[n + "CancelAnimationFrame"] || window[n + "CancelRequestAnimationFrame"]
			}
			if (/iP(ad|hone|od).*OS 6/.test(window.navigator.userAgent) || !window.requestAnimationFrame || !window.cancelAnimationFrame) {
				var i = 0;
				window.requestAnimationFrame = function(e) {
					var t = Date.now(),
					n = Math.max(i + 16, t);
					return setTimeout(function() {
						e(i = n)
					},
					n - t)
				},
				window.cancelAnimationFrame = clearTimeout
			}
		} (), i.support.touch = "ontouchstart" in document || e.DocumentTouch && document instanceof e.DocumentTouch || e.navigator.msPointerEnabled && e.navigator.msMaxTouchPoints > 0 || e.navigator.pointerEnabled && e.navigator.maxTouchPoints > 0 || !1, i.support.mutationobserver = e.MutationObserver || e.WebKitMutationObserver || null, i.Utils = {},
		i.Utils.isFullscreen = function() {
			return document.webkitFullscreenElement || document.mozFullScreenElement || document.msFullscreenElement || document.fullscreenElement || !1
		},
		i.Utils.str2json = function(e, t) {
			try {
				return t ? JSON.parse(e.replace(/([\$\w]+)\s*:/g,
				function(e, t) {
					return '"' + t + '":'
				}).replace(/'([^']+)'/g,
				function(e, t) {
					return '"' + t + '"'
				})) : new Function("", "var json = " + e + "; return JSON.parse(JSON.stringify(json));")()
			} catch(e) {
				return ! 1
			}
		},
		i.Utils.debounce = function(e, t, n) {
			var i;
			return function() {
				var r = this,
				o = arguments,
				a = n && !i;
				clearTimeout(i),
				i = setTimeout(function() {
					i = null,
					n || e.apply(r, o)
				},
				t),
				a && e.apply(r, o)
			}
		},
		i.Utils.throttle = function(e, t) {
			var n = !1;
			return function() {
				n || (e.call(), n = !0, setTimeout(function() {
					n = !1
				},
				t))
			}
		},
		i.Utils.removeCssRules = function(e) {
			var t, n, i, r, o, a, s, l, c, u;
			e && setTimeout(function() {
				try {
					for (u = document.styleSheets, r = 0, s = u.length; r < s; r++) {
						for (i = u[r], n = [], i.cssRules = i.cssRules, t = o = 0, l = i.cssRules.length; o < l; t = ++o) i.cssRules[t].type === CSSRule.STYLE_RULE && e.test(i.cssRules[t].selectorText) && n.unshift(t);
						for (a = 0, c = n.length; a < c; a++) i.deleteRule(n[a])
					}
				} catch(e) {}
			},
			0)
		},
		i.Utils.isInView = function(e, n) {
			var r = t(e);
			if (!r.is(":visible")) return ! 1;
			var o = i.$win.scrollLeft(),
			a = i.$win.scrollTop(),
			s = r.offset(),
			l = s.left,
			c = s.top;
			return n = t.extend({
				topoffset: 0,
				leftoffset: 0
			},
			n),
			c + r.height() >= a && c - n.topoffset <= a + i.$win.height() && l + r.width() >= o && l - n.leftoffset <= o + i.$win.width()
		},
		i.Utils.checkDisplay = function(e, n) {
			var r = i.$("[data-uk-margin], [data-uk-grid-match], [data-uk-grid-margin], [data-uk-check-display]", e || document);
			return e && !r.length && (r = t(e)),
			r.trigger("display.uk.check"),
			n && ("string" != typeof n && (n = '[class*="uk-animation-"]'), r.find(n).each(function() {
				var e = i.$(this),
				t = e.attr("class").match(/uk-animation-(.+)/);
				e.removeClass(t[0]).width(),
				e.addClass(t[0])
			})),
			r
		},
		i.Utils.options = function(e) {
			if ("string" != t.type(e)) return e; - 1 != e.indexOf(":") && "}" != e.trim().substr( - 1) && (e = "{" + e + "}");
			var n = e ? e.indexOf("{") : -1,
			r = {};
			if ( - 1 != n) try {
				r = i.Utils.str2json(e.substr(n))
			} catch(e) {}
			return r
		},
		i.Utils.animate = function(e, n) {
			var r = t.Deferred();
			return (e = i.$(e)).css("display", "none").addClass(n).one(i.support.animation.end,
			function() {
				e.removeClass(n),
				r.resolve()
			}),
			e.css("display", ""),
			r.promise()
		},
		i.Utils.uid = function(e) {
			return (e || "id") + (new Date).getTime() + "RAND" + Math.ceil(1e5 * Math.random())
		},
		i.Utils.template = function(e, t) {
			for (var n, i, r, o, a = e.replace(/\n/g, "\\n").replace(/\{\{\{\s*(.+?)\s*\}\}\}/g, "{{!$1}}").split(/(\{\{\s*(.+?)\s*\}\})/g), s = 0, l = [], c = 0; s < a.length;) {
				if ((n = a[s]).match(/\{\{\s*(.+?)\s*\}\}/)) switch (s += 1, n = a[s], i = n[0], r = n.substring(n.match(/^(\^|\#|\!|\~|\:)/) ? 1 : 0), i) {
				case "~":
					l.push("for(var $i=0;$i<" + r + ".length;$i++) { var $item = " + r + "[$i];"),
					c++;
					break;
				case ":":
					l.push("for(var $key in " + r + ") { var $val = " + r + "[$key];"),
					c++;
					break;
				case "#":
					l.push("if(" + r + ") {"),
					c++;
					break;
				case "^":
					l.push("if(!" + r + ") {"),
					c++;
					break;
				case "/":
					l.push("}"),
					c--;
					break;
				case "!":
					l.push("__ret.push(" + r + ");");
					break;
				default:
					l.push("__ret.push(escape(" + r + "));")
				} else l.push("__ret.push('" + n.replace(/\'/g, "\\'") + "');");
				s += 1
			}
			return o = new Function("$data", ["var __ret = [];", "try {", "with($data){", c ? '__ret = ["Not all blocks are closed correctly."]': l.join(""), "};", "}catch(e){__ret = [e.message];}", 'return __ret.join("").replace(/\\n\\n/g, "\\n");', "function escape(html) { return String(html).replace(/&/g, '&amp;').replace(/\"/g, '&quot;').replace(/</g, '&lt;').replace(/>/g, '&gt;');}"].join("\n")),
			t ? o(t) : o
		},
		i.Utils.focus = function(e, n) {
			if (! (e = t(e)).length) return e;
			var i, r = e.find("[autofocus]:first");
			return r.length ? r.focus() : (r = e.find(":input" + (n && "," + n || "")).first()).length ? r.focus() : (e.attr("tabindex") || (i = 1e3, e.attr("tabindex", i)), e[0].focus(), i && e.attr("tabindex", ""), e)
		},
		i.Utils.events = {},
		i.Utils.events.click = i.support.touch ? "tap": "click", e.UIkit = i, i.fn = function(e, n) {
			var r = arguments,
			o = e.match(/^([a-z\-]+)(?:\.([a-z]+))?/i),
			a = o[1],
			s = o[2];
			return i[a] ? this.each(function() {
				var e = t(this),
				o = e.data(a);
				o || e.data(a, o = i[a](this, s ? void 0 : n)),
				s && o[s].apply(o, Array.prototype.slice.call(r, 1))
			}) : (t.error("UIkit component [" + a + "] does not exist."), this)
		},
		t.UIkit = i, t.fn.uk = i.fn, i.langdirection = "rtl" == i.$html.attr("dir") ? "right": "left", i.components = {},
		i.component = function(e, n) {
			var r = function(n, o) {
				var a = this;
				return this.UIkit = i,
				this.element = n ? i.$(n) : null,
				this.options = t.extend(!0, {},
				this.defaults, o),
				this.plugins = {},
				this.element && this.element.data(e, this),
				this.init(),
				(this.options.plugins.length ? this.options.plugins: Object.keys(r.plugins)).forEach(function(e) {
					r.plugins[e].init && (r.plugins[e].init(a), a.plugins[e] = !0)
				}),
				this.trigger("init.uk.component", [e, this]),
				this
			};
			return r.plugins = {},
			t.extend(!0, r.prototype, {
				defaults: {
					plugins: []
				},
				boot: function() {},
				init: function() {},
				on: function(e, t, n) {
					return i.$(this.element || this).on(e, t, n)
				},
				one: function(e, t, n) {
					return i.$(this.element || this).one(e, t, n)
				},
				off: function(e) {
					return i.$(this.element || this).off(e)
				},
				trigger: function(e, t) {
					return i.$(this.element || this).trigger(e, t)
				},
				find: function(e) {
					return i.$(this.element ? this.element: []).find(e)
				},
				proxy: function(e, t) {
					var n = this;
					t.split(" ").forEach(function(t) {
						n[t] || (n[t] = function() {
							return e[t].apply(e, arguments)
						})
					})
				},
				mixin: function(e, t) {
					var n = this;
					t.split(" ").forEach(function(t) {
						n[t] || (n[t] = e[t].bind(n))
					})
				},
				option: function() {
					if (1 == arguments.length) return this.options[arguments[0]] || void 0;
					2 == arguments.length && (this.options[arguments[0]] = arguments[1])
				}
			},
			n),
			this.components[e] = r,
			this[e] = function() {
				var n, r;
				if (arguments.length) switch (arguments.length) {
				case 1:
					"string" == typeof arguments[0] || arguments[0].nodeType || arguments[0] instanceof jQuery ? n = t(arguments[0]) : r = arguments[0];
					break;
				case 2:
					n = t(arguments[0]),
					r = arguments[1]
				}
				return n && n.data(e) ? n.data(e) : new i.components[e](n, r)
			},
			i.domready && i.component.boot(e),
			r
		},
		i.plugin = function(e, t, n) {
			this.components[e].plugins[t] = n
		},
		i.component.boot = function(e) {
			i.components[e].prototype && i.components[e].prototype.boot && !i.components[e].booted && (i.components[e].prototype.boot.apply(i, []), i.components[e].booted = !0)
		},
		i.component.bootComponents = function() {
			for (var e in i.components) i.component.boot(e)
		},
		i.domObservers = [], i.domready = !1, i.ready = function(e) {
			i.domObservers.push(e),
			i.domready && e(document)
		},
		i.on = function(e, t, n) {
			return e && e.indexOf("ready.uk.dom") > -1 && i.domready && t.apply(i.$doc),
			i.$doc.on(e, t, n)
		},
		i.one = function(e, t, n) {
			return e && e.indexOf("ready.uk.dom") > -1 && i.domready ? (t.apply(i.$doc), i.$doc) : i.$doc.one(e, t, n)
		},
		i.trigger = function(e, t) {
			return i.$doc.trigger(e, t)
		},
		i.domObserve = function(e, t) {
			i.support.mutationobserver && (t = t ||
			function() {},
			i.$(e).each(function() {
				var e = this,
				n = i.$(e);
				if (!n.data("observer")) try {
					var r = new i.support.mutationobserver(i.Utils.debounce(function(i) {
						t.apply(e, [n]),
						n.trigger("changed.uk.dom")
					},
					50), {
						childList: !0,
						subtree: !0
					});
					r.observe(e, {
						childList: !0,
						subtree: !0
					}),
					n.data("observer", r)
				} catch(e) {}
			}))
		},
		i.init = function(e) {
			e = e || document,
			i.domObservers.forEach(function(t) {
				t(e)
			})
		},
		i.on("domready.uk.dom",
		function() {
			i.init(),
			i.domready && i.Utils.checkDisplay()
		}), document.addEventListener("DOMContentLoaded",
		function() {
			var e = function() {
				i.$body = i.$("body"),
				i.trigger("beforeready.uk.dom"),
				i.component.bootComponents();
				var e = requestAnimationFrame(function() {
					var t = {
						dir: {
							x: 0,
							y: 0
						},
						x: window.pageXOffset,
						y: window.pageYOffset
					},
					n = function() {
						var r = window.pageXOffset,
						o = window.pageYOffset;
						t.x == r && t.y == o || (r != t.x ? t.dir.x = r > t.x ? 1 : -1 : t.dir.x = 0, o != t.y ? t.dir.y = o > t.y ? 1 : -1 : t.dir.y = 0, t.x = r, t.y = o, i.$doc.trigger("scrolling.uk.document", [{
							dir: {
								x: t.dir.x,
								y: t.dir.y
							},
							x: r,
							y: o
						}])),
						cancelAnimationFrame(e),
						e = requestAnimationFrame(n)
					};
					return i.support.touch && i.$html.on("touchmove touchend MSPointerMove MSPointerUp pointermove pointerup", n),
					(t.x || t.y) && n(),
					n
				} ());
				if (i.trigger("domready.uk.dom"), i.support.touch && navigator.userAgent.match(/(iPad|iPhone|iPod)/g) && i.$win.on("load orientationchange resize", i.Utils.debounce(function() {
					var e = function() {
						return t(".uk-height-viewport").css("height", window.innerHeight),
						e
					};
					return e()
				} (), 100)), i.trigger("afterready.uk.dom"), i.domready = !0, i.support.mutationobserver) {
					var n = i.Utils.debounce(function() {
						requestAnimationFrame(function() {
							i.init(document.body)
						})
					},
					10);
					new i.support.mutationobserver(function(e) {
						var t = !1;
						e.every(function(e) {
							if ("childList" != e.type) return ! 0;
							for (var n, i = 0; i < e.addedNodes.length; ++i) if ((n = e.addedNodes[i]).outerHTML && -1 !== n.outerHTML.indexOf("data-uk-")) return (t = !0) && !1;
							return ! 0
						}),
						t && n()
					}).observe(document.body, {
						childList: !0,
						subtree: !0
					})
				}
			};
			return "complete" != document.readyState && "interactive" != document.readyState || setTimeout(e),
			e
		} ()), i.$html.addClass(i.support.touch ? "uk-touch": "uk-notouch"), i.support.touch) {
			var o, a = !1,
			s = ".uk-overlay, .uk-overlay-hover, .uk-overlay-toggle, .uk-animation-hover, .uk-has-hover";
			i.$html.on("mouseenter touchstart MSPointerDown pointerdown", s,
			function() {
				a && t(".uk-hover").removeClass("uk-hover"),
				a = t(this).addClass("uk-hover")
			}).on("mouseleave touchend MSPointerUp pointerup",
			function(e) {
				o = t(e.target).parents(s),
				a && a.not(o).removeClass("uk-hover")
			})
		}
		return i
	}),
	function(e) {
		function t(e, t, n, i) {
			return Math.abs(e - t) >= Math.abs(n - i) ? e - t > 0 ? "Left": "Right": n - i > 0 ? "Up": "Down"
		}
		function n() {
			c = null,
			d.last && (void 0 !== d.el && d.el.trigger("longTap"), d = {})
		}
		function i() {
			c && clearTimeout(c),
			c = null
		}
		function r() {
			a && clearTimeout(a),
			s && clearTimeout(s),
			l && clearTimeout(l),
			c && clearTimeout(c),
			a = s = l = c = null,
			d = {}
		}
		function o(e) {
			return e.pointerType == e.MSPOINTER_TYPE_TOUCH && e.isPrimary
		}
		if (!e.fn.swipeLeft) {
			var a, s, l, c, u, d = {};
			e(function() {
				var h, f, p, m = 0,
				g = 0;
				"MSGesture" in window && ((u = new MSGesture).target = document.body),
				e(document).on("MSGestureEnd gestureend",
				function(e) {
					var t = e.originalEvent.velocityX > 1 ? "Right": e.originalEvent.velocityX < -1 ? "Left": e.originalEvent.velocityY > 1 ? "Down": e.originalEvent.velocityY < -1 ? "Up": null;
					t && void 0 !== d.el && (d.el.trigger("swipe"), d.el.trigger("swipe" + t))
				}).on("touchstart MSPointerDown pointerdown",
				function(t) { ("MSPointerDown" != t.type || o(t.originalEvent)) && (p = "MSPointerDown" == t.type || "pointerdown" == t.type ? t: t.originalEvent.touches[0], h = Date.now(), f = h - (d.last || h), d.el = e("tagName" in p.target ? p.target: p.target.parentNode), a && clearTimeout(a), d.x1 = p.pageX, d.y1 = p.pageY, f > 0 && f <= 250 && (d.isDoubleTap = !0), d.last = h, c = setTimeout(n, 750), t.originalEvent && t.originalEvent.pointerId && u && ("MSPointerDown" == t.type || "pointerdown" == t.type || "touchstart" == t.type) && u.addPointer(t.originalEvent.pointerId))
				}).on("touchmove MSPointerMove pointermove",
				function(e) { ("MSPointerMove" != e.type || o(e.originalEvent)) && (p = "MSPointerMove" == e.type || "pointermove" == e.type ? e: e.originalEvent.touches[0], i(), d.x2 = p.pageX, d.y2 = p.pageY, m += Math.abs(d.x1 - d.x2), g += Math.abs(d.y1 - d.y2))
				}).on("touchend MSPointerUp pointerup",
				function(n) { ("MSPointerUp" != n.type || o(n.originalEvent)) && (i(), d.x2 && Math.abs(d.x1 - d.x2) > 30 || d.y2 && Math.abs(d.y1 - d.y2) > 30 ? l = setTimeout(function() {
						void 0 !== d.el && (d.el.trigger("swipe"), d.el.trigger("swipe" + t(d.x1, d.x2, d.y1, d.y2))),
						d = {}
					},
					0) : "last" in d && (isNaN(m) || m < 30 && g < 30 ? s = setTimeout(function() {
						var t = e.Event("tap");
						t.cancelTouch = r,
						void 0 !== d.el && d.el.trigger(t),
						d.isDoubleTap ? (void 0 !== d.el && d.el.trigger("doubleTap"), d = {}) : a = setTimeout(function() {
							a = null,
							void 0 !== d.el && d.el.trigger("singleTap"),
							d = {}
						},
						250)
					},
					0) : d = {},
					m = g = 0))
				}).on("touchcancel MSPointerCancel pointercancel", r),
				e(window).on("scroll", r)
			}),
			["swipe", "swipeLeft", "swipeRight", "swipeUp", "swipeDown", "doubleTap", "tap", "singleTap", "longTap"].forEach(function(t) {
				e.fn[t] = function(n) {
					return e(this).on(t, n)
				}
			})
		}
	} (jQuery),
	function(e) {
		"use strict";
		var t = [];
		e.component("stackMargin", {
			defaults: {
				cls: "uk-margin-small-top",
				rowfirst: !1,
				observe: !1
			},
			boot: function() {
				e.ready(function(t) {
					e.$("[data-uk-margin]", t).each(function() {
						var t = e.$(this);
						t.data("stackMargin") || e.stackMargin(t, e.Utils.options(t.attr("data-uk-margin")))
					})
				})
			},
			init: function() {
				var n = this;
				e.$win.on("resize orientationchange",
				function() {
					var t = function() {
						n.process()
					};
					return e.$(function() {
						t(),
						e.$win.on("load", t)
					}),
					e.Utils.debounce(t, 20)
				} ()),
				this.on("display.uk.check",
				function(e) {
					this.element.is(":visible") && this.process()
				}.bind(this)),
				this.options.observe && e.domObserve(this.element,
				function(e) {
					n.element.is(":visible") && n.process()
				}),
				t.push(this)
			},
			process: function() {
				var t = this.element.children();
				if (e.Utils.stackMargin(t, this.options), !this.options.rowfirst || !t.length) return this;
				var n = {},
				i = !1;
				return t.removeClass(this.options.rowfirst).each(function(t, r) {
					r = e.$(this),
					"none" != this.style.display && (t = r.offset().left, ((n[t] = n[t] || []) && n[t]).push(this), i = !1 === i ? t: Math.min(i, t))
				}),
				e.$(n[i]).addClass(this.options.rowfirst),
				this
			}
		}),
		function() {
			var t = [],
			n = function(e) {
				if (e.is(":visible")) {
					var t = e.parent().width(),
					n = e.data("width"),
					i = t / n,
					r = Math.floor(i * e.data("height"));
					e.css({
						height: t < n ? r: e.data("height")
					})
				}
			};
			e.component("responsiveElement", {
				defaults: {},
				boot: function() {
					e.ready(function(t) {
						e.$("iframe.uk-responsive-width, [data-uk-responsive]", t).each(function() {
							var t = e.$(this);
							t.data("responsiveElement") || e.responsiveElement(t, {})
						})
					})
				},
				init: function() {
					var e = this.element;
					e.attr("width") && e.attr("height") && (e.data({
						width: e.attr("width"),
						height: e.attr("height")
					}).on("display.uk.check",
					function() {
						n(e)
					}), n(e), t.push(e))
				}
			}),
			e.$win.on("resize load", e.Utils.debounce(function() {
				t.forEach(function(e) {
					n(e)
				})
			},
			15))
		} (),
		e.Utils.stackMargin = function(t, n) {
			n = e.$.extend({
				cls: "uk-margin-small-top"
			},
			n);
			var i = !1; (t = e.$(t).removeClass(n.cls)).each(function(t, n, r, o) {
				"none" != (o = e.$(this)).css("display") && (t = o.offset(), n = o.outerHeight(), r = t.top + n, o.data({
					ukMarginPos: r,
					ukMarginTop: t.top
				}), (!1 === i || t.top < i.top) && (i = {
					top: t.top,
					left: t.left,
					pos: r
				}))
			}).each(function(t) {
				"none" != (t = e.$(this)).css("display") && t.data("ukMarginTop") > i.top && t.data("ukMarginPos") > i.pos && t.addClass(n.cls)
			})
		},
		e.Utils.matchHeights = function(t, n) {
			t = e.$(t).css("min-height", "");
			var i = function(t) {
				if (! (t.length < 2)) {
					var n = 0;
					t.each(function() {
						n = Math.max(n, e.$(this).outerHeight())
					}).each(function() {
						var t = e.$(this),
						i = n - ("border-box" == t.css("box-sizing") ? 0 : t.outerHeight() - t.height());
						t.css("min-height", i + "px")
					})
				}
			}; (n = e.$.extend({
				row: !0
			},
			n)).row ? (t.first().width(), setTimeout(function() {
				var n = !1,
				r = [];
				t.each(function() {
					var t = e.$(this),
					o = t.offset().top;
					o != n && r.length && (i(e.$(r)), r = [], o = t.offset().top),
					r.push(t),
					n = o
				}),
				r.length && i(e.$(r))
			},
			0)) : i(t)
		},
		function(t) {
			e.Utils.inlineSvg = function(n, i) {
				e.$(n || 'img[src$=".svg"]', i || document).each(function() {
					var n = e.$(this),
					i = n.attr("src");
					if (!t[i]) {
						var r = e.$.Deferred();
						e.$.get(i, {
							nc: Math.random()
						},
						function(t) {
							r.resolve(e.$(t).find("svg"))
						}),
						t[i] = r.promise()
					}
					t[i].then(function(t) {
						var i = e.$(t).clone();
						n.attr("id") && i.attr("id", n.attr("id")),
						n.attr("class") && i.attr("class", n.attr("class")),
						n.attr("style") && i.attr("style", n.attr("style")),
						n.attr("width") && (i.attr("width", n.attr("width")), n.attr("height") || i.removeAttr("height")),
						n.attr("height") && (i.attr("height", n.attr("height")), n.attr("width") || i.removeAttr("width")),
						n.replaceWith(i)
					})
				})
			},
			e.ready(function(t) {
				e.Utils.inlineSvg("[data-uk-svg]", t)
			})
		} ({}),
		e.Utils.getCssVar = function(e) {
			var t, n = document.documentElement,
			i = n.appendChild(document.createElement("div"));
			i.classList.add("var-" + e);
			try {
				t = JSON.parse(t = getComputedStyle(i, ":before").content.replace(/^["'](.*)["']$/, "$1"))
			} catch(e) {
				t = void 0
			}
			return n.removeChild(i),
			t
		}
	} (UIkit),
	function(e) {
		"use strict";
		function t(t, n) {
			n = e.$.extend({
				duration: 1e3,
				transition: "easeOutExpo",
				offset: 0,
				complete: function() {}
			},
			n);
			var i = t.offset().top - n.offset,
			r = e.$doc.height(),
			o = window.innerHeight;
			i + o > r && (i = r - o),
			e.$("html,body").stop().animate({
				scrollTop: i
			},
			n.duration, n.transition).promise().done(n.complete)
		}
		e.component("smoothScroll", {
			boot: function() {
				e.$html.on("click.smooth-scroll.uikit", "[data-uk-smooth-scroll]",
				function(t) {
					var n = e.$(this);
					if (!n.data("smoothScroll")) {
						e.smoothScroll(n, e.Utils.options(n.attr("data-uk-smooth-scroll")));
						n.trigger("click")
					}
					return ! 1
				})
			},
			init: function() {
				var n = this;
				this.on("click",
				function(i) {
					i.preventDefault(),
					t(e.$(this.hash).length ? e.$(this.hash) : e.$("body"), n.options)
				})
			}
		}),
		e.Utils.scrollToElement = t,
		e.$.easing.easeOutExpo || (e.$.easing.easeOutExpo = function(e, t, n, i, r) {
			return t == r ? n + i: i * (1 - Math.pow(2, -10 * t / r)) + n
		})
	} (UIkit),
	function(e) {
		"use strict";
		var t = e.$win,
		n = e.$doc,
		i = [],
		r = function() {
			for (var e = 0; e < i.length; e++) window.requestAnimationFrame.apply(window, [i[e].check])
		};
		e.component("scrollspy", {
			defaults: {
				target: !1,
				cls: "uk-scrollspy-inview",
				initcls: "uk-scrollspy-init-inview",
				topoffset: 0,
				leftoffset: 0,
				repeat: !1,
				delay: 0
			},
			boot: function() {
				n.on("scrolling.uk.document", r),
				t.on("load resize orientationchange", e.Utils.debounce(r, 50)),
				e.ready(function(t) {
					e.$("[data-uk-scrollspy]", t).each(function() {
						var t = e.$(this);
						if (!t.data("scrollspy")) e.scrollspy(t, e.Utils.options(t.attr("data-uk-scrollspy")))
					})
				})
			},
			init: function() {
				var t, n = this,
				r = this.options.cls.split(/,/),
				o = function() {
					var i = n.options.target ? n.element.find(n.options.target) : n.element,
					o = 1 === i.length ? 1 : 0,
					a = 0;
					i.each(function(i) {
						var s = e.$(this),
						l = s.data("inviewstate"),
						c = e.Utils.isInView(s, n.options),
						u = s.data("ukScrollspyCls") || r[a].trim(); ! c || l || s.data("scrollspy-idle") || (t || (s.addClass(n.options.initcls), n.offset = s.offset(), t = !0, s.trigger("init.uk.scrollspy")), s.data("scrollspy-idle", setTimeout(function() {
							s.addClass("uk-scrollspy-inview").toggleClass(u).width(),
							s.trigger("inview.uk.scrollspy"),
							s.data("scrollspy-idle", !1),
							s.data("inviewstate", !0)
						},
						n.options.delay * o)), o++),
						!c && l && n.options.repeat && (s.data("scrollspy-idle") && (clearTimeout(s.data("scrollspy-idle")), s.data("scrollspy-idle", !1)), s.removeClass("uk-scrollspy-inview").toggleClass(u), s.data("inviewstate", !1), s.trigger("outview.uk.scrollspy")),
						a = r[a + 1] ? a + 1 : 0
					})
				};
				o(),
				this.check = o,
				i.push(this)
			}
		});
		var o = [],
		a = function() {
			for (var e = 0; e < o.length; e++) window.requestAnimationFrame.apply(window, [o[e].check])
		};
		e.component("scrollspynav", {
			defaults: {
				cls: "uk-active",
				closest: !1,
				topoffset: 0,
				leftoffset: 0,
				smoothscroll: !1
			},
			boot: function() {
				n.on("scrolling.uk.document", a),
				t.on("resize orientationchange", e.Utils.debounce(a, 50)),
				e.ready(function(t) {
					e.$("[data-uk-scrollspy-nav]", t).each(function() {
						var t = e.$(this);
						if (!t.data("scrollspynav")) e.scrollspynav(t, e.Utils.options(t.attr("data-uk-scrollspy-nav")))
					})
				})
			},
			init: function() {
				var n, i = [],
				r = this.find("a[href^='#']").each(function() {
					"#" !== this.getAttribute("href").trim() && i.push(this.getAttribute("href"))
				}),
				a = e.$(i.join(",")),
				s = this.options.cls,
				l = this.options.closest || this.options.closest,
				c = this,
				u = function() {
					n = [];
					for (var i = 0; i < a.length; i++) e.Utils.isInView(a.eq(i), c.options) && n.push(a.eq(i));
					if (n.length) {
						var o, u = t.scrollTop(),
						d = function() {
							for (var e = 0; e < n.length; e++) if (n[e].offset().top - c.options.topoffset >= u) return n[e]
						} ();
						if (!d) return;
						c.options.closest ? (r.blur().closest(l).removeClass(s), o = r.filter("a[href='#" + d.attr("id") + "']").closest(l).addClass(s)) : o = r.removeClass(s).filter("a[href='#" + d.attr("id") + "']").addClass(s),
						c.element.trigger("inview.uk.scrollspynav", [d, o])
					}
				};
				this.options.smoothscroll && e.smoothScroll && r.each(function() {
					e.smoothScroll(this, c.options.smoothscroll)
				}),
				u(),
				this.element.data("scrollspynav", this),
				this.check = u,
				o.push(this)
			}
		})
	} (UIkit),
	function(e) {
		"use strict";
		var t = [];
		e.component("toggle", {
			defaults: {
				target: !1,
				cls: "uk-hidden",
				animation: !1,
				duration: 200
			},
			boot: function() {
				e.ready(function(n) {
					e.$("[data-uk-toggle]", n).each(function() {
						var t = e.$(this);
						if (!t.data("toggle")) e.toggle(t, e.Utils.options(t.attr("data-uk-toggle")))
					}),
					setTimeout(function() {
						t.forEach(function(e) {
							e.getToggles()
						})
					},
					0)
				})
			},
			init: function() {
				var e = this;
				this.aria = -1 !== this.options.cls.indexOf("uk-hidden"),
				this.on("click",
				function(t) {
					e.element.is('a[href="#"]') && t.preventDefault(),
					e.toggle()
				}),
				t.push(this)
			},
			toggle: function() {
				if (this.getToggles(), this.totoggle.length) {
					if (this.options.animation && e.support.animation) {
						var t = this,
						n = this.options.animation.split(",");
						1 == n.length && (n[1] = n[0]),
						n[0] = n[0].trim(),
						n[1] = n[1].trim(),
						this.totoggle.css("animation-duration", this.options.duration + "ms"),
						this.totoggle.each(function() {
							var i = e.$(this);
							i.hasClass(t.options.cls) ? (i.toggleClass(t.options.cls), e.Utils.animate(i, n[0]).then(function() {
								i.css("animation-duration", ""),
								e.Utils.checkDisplay(i)
							})) : e.Utils.animate(this, n[1] + " uk-animation-reverse").then(function() {
								i.toggleClass(t.options.cls).css("animation-duration", ""),
								e.Utils.checkDisplay(i)
							})
						})
					} else this.totoggle.toggleClass(this.options.cls),
					e.Utils.checkDisplay(this.totoggle);
					this.updateAria()
				}
			},
			getToggles: function() {
				this.totoggle = this.options.target ? e.$(this.options.target) : [],
				this.updateAria()
			},
			updateAria: function() {
				this.aria && this.totoggle.length && this.totoggle.not("[aria-hidden]").each(function() {
					e.$(this).attr("aria-hidden", e.$(this).hasClass("uk-hidden"))
				})
			}
		})
	} (UIkit),
	function(e) {
		"use strict";
		e.component("alert", {
			defaults: {
				fade: !0,
				duration: 200,
				trigger: ".uk-alert-close"
			},
			boot: function() {
				e.$html.on("click.alert.uikit", "[data-uk-alert]",
				function(t) {
					var n = e.$(this);
					if (!n.data("alert")) {
						var i = e.alert(n, e.Utils.options(n.attr("data-uk-alert")));
						e.$(t.target).is(i.options.trigger) && (t.preventDefault(), i.close())
					}
				})
			},
			init: function() {
				var e = this;
				this.on("click", this.options.trigger,
				function(t) {
					t.preventDefault(),
					e.close()
				})
			},
			close: function() {
				var e = this.trigger("close.uk.alert"),
				t = function() {
					this.trigger("closed.uk.alert").remove()
				}.bind(this);
				this.options.fade ? e.css("overflow", "hidden").css("max-height", e.height()).animate({
					height: 0,
					opacity: 0,
					paddingTop: 0,
					paddingBottom: 0,
					marginTop: 0,
					marginBottom: 0
				},
				this.options.duration, t) : t()
			}
		})
	} (UIkit),
	function(e) {
		"use strict";
		e.component("buttonRadio", {
			defaults: {
				activeClass: "uk-active",
				target: ".uk-button"
			},
			boot: function() {
				e.$html.on("click.buttonradio.uikit", "[data-uk-button-radio]",
				function(t) {
					var n = e.$(this);
					if (!n.data("buttonRadio")) {
						var i = e.buttonRadio(n, e.Utils.options(n.attr("data-uk-button-radio"))),
						r = e.$(t.target);
						r.is(i.options.target) && r.trigger("click")
					}
				})
			},
			init: function() {
				var t = this;
				this.find(t.options.target).attr("aria-checked", "false").filter("." + t.options.activeClass).attr("aria-checked", "true"),
				this.on("click", this.options.target,
				function(n) {
					var i = e.$(this);
					i.is('a[href="#"]') && n.preventDefault(),
					t.find(t.options.target).not(i).removeClass(t.options.activeClass).blur(),
					i.addClass(t.options.activeClass),
					t.find(t.options.target).not(i).attr("aria-checked", "false"),
					i.attr("aria-checked", "true"),
					t.trigger("change.uk.button", [i])
				})
			},
			getSelected: function() {
				return this.find("." + this.options.activeClass)
			}
		}),
		e.component("buttonCheckbox", {
			defaults: {
				activeClass: "uk-active",
				target: ".uk-button"
			},
			boot: function() {
				e.$html.on("click.buttoncheckbox.uikit", "[data-uk-button-checkbox]",
				function(t) {
					var n = e.$(this);
					if (!n.data("buttonCheckbox")) {
						var i = e.buttonCheckbox(n, e.Utils.options(n.attr("data-uk-button-checkbox"))),
						r = e.$(t.target);
						r.is(i.options.target) && r.trigger("click")
					}
				})
			},
			init: function() {
				var t = this;
				this.find(t.options.target).attr("aria-checked", "false").filter("." + t.options.activeClass).attr("aria-checked", "true"),
				this.on("click", this.options.target,
				function(n) {
					var i = e.$(this);
					i.is('a[href="#"]') && n.preventDefault(),
					i.toggleClass(t.options.activeClass).blur(),
					i.attr("aria-checked", i.hasClass(t.options.activeClass)),
					t.trigger("change.uk.button", [i])
				})
			},
			getSelected: function() {
				return this.find("." + this.options.activeClass)
			}
		}),
		e.component("button", {
			defaults: {},
			boot: function() {
				e.$html.on("click.button.uikit", "[data-uk-button]",
				function(t) {
					var n = e.$(this);
					if (!n.data("button")) {
						e.button(n, e.Utils.options(n.attr("data-uk-button")));
						n.trigger("click")
					}
				})
			},
			init: function() {
				var e = this;
				this.element.attr("aria-pressed", this.element.hasClass("uk-active")),
				this.on("click",
				function(t) {
					e.element.is('a[href="#"]') && t.preventDefault(),
					e.toggle(),
					e.trigger("change.uk.button", [e.element.blur().hasClass("uk-active")])
				})
			},
			toggle: function() {
				this.element.toggleClass("uk-active"),
				this.element.attr("aria-pressed", this.element.hasClass("uk-active"))
			}
		})
	} (UIkit),
	function(e) {
		"use strict";
		function t(t, n, i, r) {
			if (t = e.$(t), n = e.$(n), i = i || window.innerWidth, r = r || t.offset(), n.length) {
				var o = n.outerWidth();
				if (t.css("min-width", o), "right" == e.langdirection) {
					var a = i - (n.offset().left + o),
					s = i - (t.offset().left + t.outerWidth());
					t.css("margin-right", a - s)
				} else t.css("margin-left", n.offset().left - r.left)
			}
		}
		var n, i = !1,
		r = {
			x: {
				"bottom-left": "bottom-right",
				"bottom-right": "bottom-left",
				"bottom-center": "bottom-center",
				"top-left": "top-right",
				"top-right": "top-left",
				"top-center": "top-center",
				"left-top": "right-top",
				"left-bottom": "right-bottom",
				"left-center": "right-center",
				"right-top": "left-top",
				"right-bottom": "left-bottom",
				"right-center": "left-center"
			},
			y: {
				"bottom-left": "top-left",
				"bottom-right": "top-right",
				"bottom-center": "top-center",
				"top-left": "bottom-left",
				"top-right": "bottom-right",
				"top-center": "bottom-center",
				"left-top": "left-bottom",
				"left-bottom": "left-top",
				"left-center": "left-center",
				"right-top": "right-bottom",
				"right-bottom": "right-top",
				"right-center": "right-center"
			},
			xy: {
				"bottom-left": "top-right",
				"bottom-right": "top-left",
				"bottom-center": "top-center",
				"top-left": "bottom-right",
				"top-right": "bottom-left",
				"top-center": "bottom-center",
				"left-top": "right-bottom",
				"left-bottom": "right-top",
				"left-center": "right-center",
				"right-top": "left-bottom",
				"right-bottom": "left-top",
				"right-center": "left-center"
			}
		};
		e.component("dropdown", {
			defaults: {
				mode: "hover",
				pos: "bottom-left",
				offset: 0,
				remaintime: 800,
				justify: !1,
				boundary: e.$win,
				delay: 0,
				dropdownSelector: ".uk-dropdown,.uk-dropdown-blank",
				hoverDelayIdle: 250,
				preventflip: !1
			},
			remainIdle: !1,
			boot: function() {
				var t = e.support.touch ? "click": "mouseenter";
				e.$html.on(t + ".dropdown.uikit focus pointerdown", "[data-uk-dropdown]",
				function(n) {
					var i = e.$(this);
					if (!i.data("dropdown")) {
						var r = e.dropdown(i, e.Utils.options(i.attr("data-uk-dropdown"))); ("click" == n.type || "mouseenter" == n.type && "hover" == r.options.mode) && r.element.trigger(t),
						r.dropdown.length && n.preventDefault()
					}
				})
			},
			init: function() {
				var t = this;
				this.dropdown = this.find(this.options.dropdownSelector),
				this.offsetParent = this.dropdown.parents().filter(function() {
					return - 1 !== e.$.inArray(e.$(this).css("position"), ["relative", "fixed", "absolute"])
				}).slice(0, 1),
				this.offsetParent.length || (this.offsetParent = this.element),
				this.centered = this.dropdown.hasClass("uk-dropdown-center"),
				this.justified = !!this.options.justify && e.$(this.options.justify),
				this.boundary = e.$(this.options.boundary),
				this.boundary.length || (this.boundary = e.$win),
				this.dropdown.hasClass("uk-dropdown-up") && (this.options.pos = "top-left"),
				this.dropdown.hasClass("uk-dropdown-flip") && (this.options.pos = this.options.pos.replace("left", "right")),
				this.dropdown.hasClass("uk-dropdown-center") && (this.options.pos = this.options.pos.replace(/(left|right)/, "center")),
				this.element.attr("aria-haspopup", "true"),
				this.element.attr("aria-expanded", this.element.hasClass("uk-open")),
				this.dropdown.attr("aria-hidden", "true"),
				"click" == this.options.mode || e.support.touch ? this.on("click.uk.dropdown",
				function(n) {
					var i = e.$(n.target);
					i.parents(t.options.dropdownSelector).length || ((i.is("a[href='#']") || i.parent().is("a[href='#']") || t.dropdown.length && !t.dropdown.is(":visible")) && n.preventDefault(), i.blur()),
					t.element.hasClass("uk-open") ? (!t.dropdown.find(n.target).length || i.is(".uk-dropdown-close") || i.parents(".uk-dropdown-close").length) && t.hide() : t.show()
				}) : this.on("mouseenter",
				function(e) {
					t.trigger("pointerenter.uk.dropdown", [t]),
					t.remainIdle && clearTimeout(t.remainIdle),
					n && clearTimeout(n),
					i && i == t || (n = i && i != t ? setTimeout(function() {
						n = setTimeout(t.show.bind(t), t.options.delay)
					},
					t.options.hoverDelayIdle) : setTimeout(t.show.bind(t), t.options.delay))
				}).on("mouseleave",
				function() {
					n && clearTimeout(n),
					t.remainIdle = setTimeout(function() {
						i && i == t && t.hide()
					},
					t.options.remaintime),
					t.trigger("pointerleave.uk.dropdown", [t])
				}).on("click",
				function(n) {
					var r = e.$(n.target);
					t.remainIdle && clearTimeout(t.remainIdle),
					i && i == t ? (!t.dropdown.find(n.target).length || r.is(".uk-dropdown-close") || r.parents(".uk-dropdown-close").length) && t.hide() : ((r.is("a[href='#']") || r.parent().is("a[href='#']")) && n.preventDefault(), t.show())
				})
			},
			show: function() {
				e.$html.off("click.outer.dropdown"),
				i && i != this && i.hide(!0),
				n && clearTimeout(n),
				this.trigger("beforeshow.uk.dropdown", [this]),
				this.checkDimensions(),
				this.element.addClass("uk-open"),
				this.element.attr("aria-expanded", "true"),
				this.dropdown.attr("aria-hidden", "false"),
				this.trigger("show.uk.dropdown", [this]),
				e.Utils.checkDisplay(this.dropdown, !0),
				e.Utils.focus(this.dropdown),
				i = this,
				this.registerOuterClick()
			},
			hide: function(e) {
				this.trigger("beforehide.uk.dropdown", [this, e]),
				this.element.removeClass("uk-open"),
				this.remainIdle && clearTimeout(this.remainIdle),
				this.remainIdle = !1,
				this.element.attr("aria-expanded", "false"),
				this.dropdown.attr("aria-hidden", "true"),
				this.trigger("hide.uk.dropdown", [this, e]),
				i == this && (i = !1)
			},
			registerOuterClick: function() {
				var t = this;
				e.$html.off("click.outer.dropdown"),
				setTimeout(function() {
					e.$html.on("click.outer.dropdown",
					function(r) {
						n && clearTimeout(n);
						e.$(r.target);
						i != t || t.element.find(r.target).length || (t.hide(!0), e.$html.off("click.outer.dropdown"))
					})
				},
				10)
			},
			checkDimensions: function() {
				if (this.dropdown.length) {
					this.dropdown.removeClass("uk-dropdown-top uk-dropdown-bottom uk-dropdown-left uk-dropdown-right uk-dropdown-stack uk-dropdown-autoflip").css({
						topLeft: "",
						left: "",
						marginLeft: "",
						marginRight: ""
					}),
					this.justified && this.justified.length && this.dropdown.css("min-width", "");
					var n, i = e.$.extend({},
					this.offsetParent.offset(), {
						width: this.offsetParent[0].offsetWidth,
						height: this.offsetParent[0].offsetHeight
					}),
					o = this.options.offset,
					a = this.dropdown,
					s = (a.show().offset(), a.outerWidth()),
					l = a.outerHeight(),
					c = this.boundary.width(),
					u = (this.boundary[0] !== window && this.boundary.offset() && this.boundary.offset(), this.options.pos),
					d = {
						"bottom-left": {
							top: 0 + i.height + o,
							left: 0
						},
						"bottom-right": {
							top: 0 + i.height + o,
							left: 0 + i.width - s
						},
						"bottom-center": {
							top: 0 + i.height + o,
							left: 0 + i.width / 2 - s / 2
						},
						"top-left": {
							top: 0 - l - o,
							left: 0
						},
						"top-right": {
							top: 0 - l - o,
							left: 0 + i.width - s
						},
						"top-center": {
							top: 0 - l - o,
							left: 0 + i.width / 2 - s / 2
						},
						"left-top": {
							top: 0,
							left: 0 - s - o
						},
						"left-bottom": {
							top: 0 + i.height - l,
							left: 0 - s - o
						},
						"left-center": {
							top: 0 + i.height / 2 - l / 2,
							left: 0 - s - o
						},
						"right-top": {
							top: 0,
							left: 0 + i.width + o
						},
						"right-bottom": {
							top: 0 + i.height - l,
							left: 0 + i.width + o
						},
						"right-center": {
							top: 0 + i.height / 2 - l / 2,
							left: 0 + i.width + o
						}
					},
					h = {};
					if (n = u.split("-"), h = d[u] ? d[u] : d["bottom-left"], this.justified && this.justified.length) t(a.css({
						left: 0
					}), this.justified, c);
					else if (!0 !== this.options.preventflip) {
						var f;
						switch (this.checkBoundary(i.left + h.left, i.top + h.top, s, l, c)) {
						case "x":
							"x" !== this.options.preventflip && (f = r.x[u] || "right-top");
							break;
						case "y":
							"y" !== this.options.preventflip && (f = r.y[u] || "top-left");
							break;
						case "xy":
							this.options.preventflip || (f = r.xy[u] || "right-bottom")
						}
						f && (n = f.split("-"), h = d[f] ? d[f] : d["bottom-left"], a.addClass("uk-dropdown-autoflip"), this.checkBoundary(i.left + h.left, i.top + h.top, s, l, c) && (n = u.split("-"), h = d[u] ? d[u] : d["bottom-left"]))
					}
					s > c && (a.addClass("uk-dropdown-stack"), this.trigger("stack.uk.dropdown", [this])),
					a.css(h).css("display", "").addClass("uk-dropdown-" + n[0])
				}
			},
			checkBoundary: function(t, n, i, r, o) {
				var a = "";
				return (t < 0 || t - e.$win.scrollLeft() + i > o) && (a += "x"),
				(n - e.$win.scrollTop() < 0 || n - e.$win.scrollTop() + r > window.innerHeight) && (a += "y"),
				a
			}
		}),
		e.component("dropdownOverlay", {
			defaults: {
				justify: !1,
				cls: "",
				duration: 200
			},
			boot: function() {
				e.ready(function(t) {
					e.$("[data-uk-dropdown-overlay]", t).each(function() {
						var t = e.$(this);
						t.data("dropdownOverlay") || e.dropdownOverlay(t, e.Utils.options(t.attr("data-uk-dropdown-overlay")))
					})
				})
			},
			init: function() {
				var n = this;
				this.justified = !!this.options.justify && e.$(this.options.justify),
				this.overlay = this.element.find("uk-dropdown-overlay"),
				this.overlay.length || (this.overlay = e.$('<div class="uk-dropdown-overlay"></div>').appendTo(this.element)),
				this.overlay.addClass(this.options.cls),
				this.on({
					"beforeshow.uk.dropdown": function(e, i) {
						n.dropdown = i,
						n.justified && n.justified.length && t(n.overlay.css({
							display: "block",
							marginLeft: "",
							marginRight: ""
						}), n.justified, n.justified.outerWidth())
					},
					"show.uk.dropdown": function(t, i) {
						var r = n.dropdown.dropdown.outerHeight(!0);
						n.dropdown.element.removeClass("uk-open"),
						n.overlay.stop().css("display", "block").animate({
							height: r
						},
						n.options.duration,
						function() {
							n.dropdown.dropdown.css("visibility", ""),
							n.dropdown.element.addClass("uk-open"),
							e.Utils.checkDisplay(n.dropdown.dropdown, !0)
						}),
						n.pointerleave = !1
					},
					"hide.uk.dropdown": function() {
						n.overlay.stop().animate({
							height: 0
						},
						n.options.duration)
					},
					"pointerenter.uk.dropdown": function(e, t) {
						clearTimeout(n.remainIdle)
					},
					"pointerleave.uk.dropdown": function(e, t) {
						n.pointerleave = !0
					}
				}),
				this.overlay.on({
					mouseenter: function() {
						n.remainIdle && (clearTimeout(n.dropdown.remainIdle), clearTimeout(n.remainIdle))
					},
					mouseleave: function() {
						n.pointerleave && i && (n.remainIdle = setTimeout(function() {
							i && i.hide()
						},
						i.options.remaintime))
					}
				})
			}
		})
	} (UIkit),
	function(e) {
		"use strict";
		var t = [];
		e.component("gridMatchHeight", {
			defaults: {
				target: !1,
				row: !0,
				ignorestacked: !1,
				observe: !1
			},
			boot: function() {
				e.ready(function(t) {
					e.$("[data-uk-grid-match]", t).each(function() {
						var t = e.$(this);
						t.data("gridMatchHeight") || e.gridMatchHeight(t, e.Utils.options(t.attr("data-uk-grid-match")))
					})
				})
			},
			init: function() {
				var n = this;
				this.columns = this.element.children(),
				this.elements = this.options.target ? this.find(this.options.target) : this.columns,
				this.columns.length && (e.$win.on("load resize orientationchange",
				function() {
					var t = function() {
						n.element.is(":visible") && n.match()
					};
					return e.$(function() {
						t()
					}),
					e.Utils.debounce(t, 50)
				} ()), this.options.observe && e.domObserve(this.element,
				function(e) {
					n.element.is(":visible") && n.match()
				}), this.on("display.uk.check",
				function(e) {
					this.element.is(":visible") && this.match()
				}.bind(this)), t.push(this))
			},
			match: function() {
				var t = this.columns.filter(":visible:first");
				if (t.length) return Math.ceil(100 * parseFloat(t.css("width")) / parseFloat(t.parent().css("width"))) >= 100 && !this.options.ignorestacked ? this.revert() : e.Utils.matchHeights(this.elements, this.options),
				this
			},
			revert: function() {
				return this.elements.css("min-height", ""),
				this
			}
		}),
		e.component("gridMargin", {
			defaults: {
				cls: "uk-grid-margin",
				rowfirst: "uk-row-first"
			},
			boot: function() {
				e.ready(function(t) {
					e.$("[data-uk-grid-margin]", t).each(function() {
						var t = e.$(this);
						t.data("gridMargin") || e.gridMargin(t, e.Utils.options(t.attr("data-uk-grid-margin")))
					})
				})
			},
			init: function() {
				e.stackMargin(this.element, this.options)
			}
		})
	} (UIkit),
	function(e) {
		"use strict";
		function t(t, n) {
			if (n) return "object" == typeof t ? (t = t instanceof jQuery ? t: e.$(t)).parent().length && (n.persist = t, n.persist.data("modalPersistParent", t.parent())) : t = "string" == typeof t || "number" == typeof t ? e.$("<div></div>").html(t) : e.$("<div></div>").html("UIkit.modal Error: Unsupported data type: " + typeof t),
			t.appendTo(n.element.find(".uk-modal-dialog")),
			n
		}
		var n, i = !1,
		r = 0,
		o = e.$html;
		e.$win.on("resize orientationchange", e.Utils.debounce(function() {
			e.$(".uk-modal.uk-open").each(function() {
				return e.$(this).data("modal") && e.$(this).data("modal").resize()
			})
		},
		150)),
		e.component("modal", {
			defaults: {
				keyboard: !0,
				bgclose: !0,
				minScrollHeight: 150,
				center: !1,
				modal: !0
			},
			scrollable: !1,
			transition: !1,
			hasTransitioned: !0,
			init: function() {
				if (n || (n = e.$("body")), this.element.length) {
					var t = this;
					this.paddingdir = "padding-" + ("left" == e.langdirection ? "right": "left"),
					this.dialog = this.find(".uk-modal-dialog"),
					this.active = !1,
					this.element.attr("aria-hidden", this.element.hasClass("uk-open")),
					this.on("click", ".uk-modal-close",
					function(e) {
						e.preventDefault(),
						t.hide()
					}).on("click",
					function(n) {
						e.$(n.target)[0] == t.element[0] && t.options.bgclose && t.hide()
					}),
					e.domObserve(this.element,
					function(e) {
						t.resize()
					})
				}
			},
			toggle: function() {
				return this[this.isActive() ? "hide": "show"]()
			},
			show: function() {
				if (this.element.length) {
					var t = this;
					if (!this.isActive()) return this.options.modal && i && i.hide(!0),
					this.element.removeClass("uk-open").show(),
					this.resize(!0),
					this.options.modal && (i = this),
					this.active = !0,
					r++,
					e.support.transition ? (this.hasTransitioned = !1, this.element.one(e.support.transition.end,
					function() {
						t.hasTransitioned = !0,
						e.Utils.focus(t.dialog, "a[href]")
					}).addClass("uk-open")) : (this.element.addClass("uk-open"), e.Utils.focus(this.dialog, "a[href]")),
					o.addClass("uk-modal-page").height(),
					this.element.attr("aria-hidden", "false"),
					this.element.trigger("show.uk.modal"),
					e.Utils.checkDisplay(this.dialog, !0),
					this
				}
			},
			hide: function(t) {
				if (!t && e.support.transition && this.hasTransitioned) {
					var n = this;
					this.one(e.support.transition.end,
					function() {
						n._hide()
					}).removeClass("uk-open")
				} else this._hide();
				return this
			},
			resize: function(e) {
				if (this.isActive() || e) {
					var t = n.width();
					if (this.scrollbarwidth = window.innerWidth - t, n.css(this.paddingdir, this.scrollbarwidth), this.element.css("overflow-y", this.scrollbarwidth ? "scroll": "auto"), !this.updateScrollable() && this.options.center) {
						var i = this.dialog.outerHeight(),
						r = parseInt(this.dialog.css("margin-top"), 10) + parseInt(this.dialog.css("margin-bottom"), 10);
						i + r < window.innerHeight ? this.dialog.css({
							top: window.innerHeight / 2 - i / 2 - r
						}) : this.dialog.css({
							top: ""
						})
					}
				}
			},
			updateScrollable: function() {
				var e = this.dialog.find(".uk-overflow-container:visible:first");
				if (e.length) {
					e.css("height", 0);
					var t = Math.abs(parseInt(this.dialog.css("margin-top"), 10)),
					n = this.dialog.outerHeight(),
					i = window.innerHeight - 2 * (t < 20 ? 20 : t) - n;
					return e.css({
						maxHeight: i < this.options.minScrollHeight ? "": i,
						height: ""
					}),
					!0
				}
				return ! 1
			},
			_hide: function() {
				this.active = !1,
				r > 0 ? r--:r = 0,
				this.element.hide().removeClass("uk-open"),
				this.element.attr("aria-hidden", "true"),
				r || (o.removeClass("uk-modal-page"), n.css(this.paddingdir, "")),
				i === this && (i = !1),
				this.trigger("hide.uk.modal")
			},
			isActive: function() {
				return this.element.hasClass("uk-open")
			}
		}),
		e.component("modalTrigger", {
			boot: function() {
				e.$html.on("click.modal.uikit", "[data-uk-modal]",
				function(t) {
					var n = e.$(this);
					n.is("a") && t.preventDefault(),
					n.data("modalTrigger") || e.modalTrigger(n, e.Utils.options(n.attr("data-uk-modal"))).show()
				}),
				e.$html.on("keydown.modal.uikit",
				function(e) {
					i && 27 === e.keyCode && i.options.keyboard && (e.preventDefault(), i.hide())
				})
			},
			init: function() {
				var t = this;
				this.options = e.$.extend({
					target: !!t.element.is("a") && t.element.attr("href")
				},
				this.options),
				this.modal = e.modal(this.options.target, this.options),
				this.on("click",
				function(e) {
					e.preventDefault(),
					t.show()
				}),
				this.proxy(this.modal, "show hide isActive")
			}
		}),
		e.modal.dialog = function(n, i) {
			var r = e.modal(e.$(e.modal.dialog.template).appendTo("body"), i);
			return r.on("hide.uk.modal",
			function() {
				r.persist && (r.persist.appendTo(r.persist.data("modalPersistParent")), r.persist = !1),
				r.element.remove()
			}),
			t(n, r),
			r
		},
		e.modal.dialog.template = '<div class="uk-modal"><div class="uk-modal-dialog" style="min-height:0;"></div></div>',
		e.modal.alert = function(t, n) {
			n = e.$.extend(!0, {
				bgclose: !1,
				keyboard: !1,
				modal: !1,
				labels: e.modal.labels
			},
			n);
			var i = e.modal.dialog(['<div class="uk-margin uk-modal-content">' + String(t) + "</div>", '<div class="uk-modal-footer uk-text-right"><button class="uk-button uk-button-primary uk-modal-close">' + n.labels.Ok + "</button></div>"].join(""), n);
			return i.on("show.uk.modal",
			function() {
				setTimeout(function() {
					i.element.find("button:first").focus()
				},
				50)
			}),
			i.show()
		},
		e.modal.confirm = function(t, n, i) {
			var r = arguments.length > 1 && arguments[arguments.length - 1] ? arguments[arguments.length - 1] : {};
			n = e.$.isFunction(n) ? n: function() {},
			i = e.$.isFunction(i) ? i: function() {},
			r = e.$.extend(!0, {
				bgclose: !1,
				keyboard: !1,
				modal: !1,
				labels: e.modal.labels
			},
			e.$.isFunction(r) ? {}: r);
			var o = e.modal.dialog(['<div class="uk-margin uk-modal-content">' + String(t) + "</div>", '<div class="uk-modal-footer uk-text-right"><button class="uk-button js-modal-confirm-cancel">' + r.labels.Cancel + '</button> <button class="uk-button uk-button-primary js-modal-confirm">' + r.labels.Ok + "</button></div>"].join(""), r);
			return o.element.find(".js-modal-confirm, .js-modal-confirm-cancel").on("click",
			function() {
				e.$(this).is(".js-modal-confirm") ? n() : i(),
				o.hide()
			}),
			o.on("show.uk.modal",
			function() {
				setTimeout(function() {
					o.element.find(".js-modal-confirm").focus()
				},
				50)
			}),
			o.show()
		},
		e.modal.prompt = function(t, n, i, r) {
			i = e.$.isFunction(i) ? i: function(e) {},
			r = e.$.extend(!0, {
				bgclose: !1,
				keyboard: !1,
				modal: !1,
				labels: e.modal.labels
			},
			r);
			var o = e.modal.dialog([t ? '<div class="uk-modal-content uk-form">' + String(t) + "</div>": "", '<div class="uk-margin-small-top uk-modal-content uk-form"><p><input type="text" class="uk-width-1-1"></p></div>', '<div class="uk-modal-footer uk-text-right"><button class="uk-button uk-modal-close">' + r.labels.Cancel + '</button> <button class="uk-button uk-button-primary js-modal-ok">' + r.labels.Ok + "</button></div>"].join(""), r),
			a = o.element.find("input[type='text']").val(n || "").on("keyup",
			function(e) {
				13 == e.keyCode && o.element.find(".js-modal-ok").trigger("click")
			});
			return o.element.find(".js-modal-ok").on("click",
			function() { ! 1 !== i(a.val()) && o.hide()
			}),
			o.show()
		},
		e.modal.blockUI = function(t, n) {
			var i = e.modal.dialog(['<div class="uk-margin uk-modal-content">' + String(t || '<div class="uk-text-center">...</div>') + "</div>"].join(""), e.$.extend({
				bgclose: !1,
				keyboard: !1,
				modal: !1
			},
			n));
			return i.content = i.element.find(".uk-modal-content:first"),
			i.show()
		},
		e.modal.labels = {
			Ok: "Ok",
			Cancel: "Cancel"
		}
	} (UIkit),
	function(e) {
		"use strict";
		function t(t) {
			var n = e.$(t),
			i = "auto";
			if (n.is(":visible")) i = n.outerHeight();
			else {
				var r = {
					position: n.css("position"),
					visibility: n.css("visibility"),
					display: n.css("display")
				};
				i = n.css({
					position: "absolute",
					visibility: "hidden",
					display: "block"
				}).outerHeight(),
				n.css(r)
			}
			return i
		}
		e.component("nav", {
			defaults: {
				toggle: '>li.uk-parent > a[href="#"]',
				lists: ">li.uk-parent > ul",
				multiple: !1
			},
			boot: function() {
				e.ready(function(t) {
					e.$("[data-uk-nav]", t).each(function() {
						var t = e.$(this);
						if (!t.data("nav")) e.nav(t, e.Utils.options(t.attr("data-uk-nav")))
					})
				})
			},
			init: function() {
				var t = this;
				this.on("click.uk.nav", this.options.toggle,
				function(n) {
					n.preventDefault();
					var i = e.$(this);
					t.open(i.parent()[0] == t.element[0] ? i: i.parent("li"))
				}),
				this.update(),
				e.domObserve(this.element,
				function(e) {
					t.element.find(t.options.lists).not("[role]").length && t.update()
				})
			},
			update: function() {
				var t = this;
				this.find(this.options.lists).each(function() {
					var n = e.$(this).attr("role", "menu"),
					i = n.closest("li"),
					r = i.hasClass("uk-active");
					i.data("list-container") || (n.wrap('<div style="overflow:hidden;height:0;position:relative;"></div>'), i.data("list-container", n.parent()[r ? "removeClass": "addClass"]("uk-hidden"))),
					i.attr("aria-expanded", i.hasClass("uk-open")),
					r && t.open(i, !0)
				})
			},
			open: function(n, i) {
				var r = this,
				o = this.element,
				a = e.$(n),
				s = a.data("list-container");
				this.options.multiple || o.children(".uk-open").not(n).each(function() {
					var t = e.$(this);
					t.data("list-container") && t.data("list-container").stop().animate({
						height: 0
					},
					function() {
						e.$(this).parent().removeClass("uk-open").end().addClass("uk-hidden")
					})
				}),
				a.toggleClass("uk-open"),
				a.attr("aria-expanded", a.hasClass("uk-open")),
				s && (a.hasClass("uk-open") && s.removeClass("uk-hidden"), i ? (s.stop().height(a.hasClass("uk-open") ? "auto": 0), a.hasClass("uk-open") || s.addClass("uk-hidden"), this.trigger("display.uk.check")) : s.stop().animate({
					height: a.hasClass("uk-open") ? t(s.find("ul:first")) : 0
				},
				function() {
					a.hasClass("uk-open") ? s.css("height", "") : s.addClass("uk-hidden"),
					r.trigger("display.uk.check")
				}))
			}
		})
	} (UIkit),
	function(e) {
		"use strict";
		var t = {
			x: window.scrollX,
			y: window.scrollY
		},
		n = (e.$win, e.$doc, e.$html),
		i = {
			show: function(i, r) {
				if ((i = e.$(i)).length) {
					r = e.$.extend({
						mode: "push"
					},
					r);
					var o = e.$("body"),
					a = i.find(".uk-offcanvas-bar:first"),
					s = "right" == e.langdirection,
					l = (a.hasClass("uk-offcanvas-bar-flip") ? -1 : 1) * (s ? -1 : 1),
					c = window.innerWidth - o.width();
					t = {
						x: window.pageXOffset,
						y: window.pageYOffset
					},
					a.attr("mode", r.mode),
					i.addClass("uk-active"),
					o.css({
						width: window.innerWidth - c,
						height: window.innerHeight
					}).addClass("uk-offcanvas-page"),
					"push" != r.mode && "reveal" != r.mode || o.css(s ? "margin-right": "margin-left", (s ? -1 : 1) * (a.outerWidth() * l)),
					"reveal" == r.mode && a.css("clip", "rect(0, " + a.outerWidth() + "px, 100vh, 0)"),
					n.css("margin-top", -1 * t.y).width(),
					a.addClass("uk-offcanvas-bar-show"),
					this._initElement(i),
					a.trigger("show.uk.offcanvas", [i, a]),
					i.attr("aria-hidden", "false")
				}
			},
			hide: function(i) {
				var r = e.$("body"),
				o = e.$(".uk-offcanvas.uk-active"),
				a = "right" == e.langdirection,
				s = o.find(".uk-offcanvas-bar:first"),
				l = function() {
					r.removeClass("uk-offcanvas-page").css({
						width: "",
						height: "",
						marginLeft: "",
						marginRight: ""
					}),
					o.removeClass("uk-active"),
					s.removeClass("uk-offcanvas-bar-show"),
					n.css("margin-top", ""),
					window.scrollTo(t.x, t.y),
					s.trigger("hide.uk.offcanvas", [o, s]),
					o.attr("aria-hidden", "true")
				};
				o.length && ("none" == s.attr("mode") && (i = !0), e.support.transition && !i ? (r.one(e.support.transition.end,
				function() {
					l()
				}).css(a ? "margin-right": "margin-left", ""), "reveal" == s.attr("mode") && s.css("clip", ""), setTimeout(function() {
					s.removeClass("uk-offcanvas-bar-show")
				},
				0)) : l())
			},
			_initElement: function(t) {
				t.data("OffcanvasInit") || (t.on("click.uk.offcanvas swipeRight.uk.offcanvas swipeLeft.uk.offcanvas",
				function(t) {
					var n = e.$(t.target);
					if (!t.type.match(/swipe/) && !n.hasClass("uk-offcanvas-close")) {
						if (n.hasClass("uk-offcanvas-bar")) return;
						if (n.parents(".uk-offcanvas-bar:first").length) return
					}
					t.stopImmediatePropagation(),
					i.hide()
				}), t.on("click", 'a[href*="#"]',
				function(t) {
					var n = e.$(this),
					r = n.attr("href");
					"#" != r && (e.$doc.one("hide.uk.offcanvas",
					function() {
						var t;
						try {
							t = e.$(n[0].hash)
						} catch(e) {
							t = ""
						}
						t.length || (t = e.$('[name="' + n[0].hash.replace("#", "") + '"]')),
						t.length && e.Utils.scrollToElement ? e.Utils.scrollToElement(t, e.Utils.options(n.attr("data-uk-smooth-scroll") || "{}")) : window.location.href = r
					}), i.hide())
				}), t.data("OffcanvasInit", !0))
			}
		};
		e.component("offcanvasTrigger", {
			boot: function() {
				n.on("click.offcanvas.uikit", "[data-uk-offcanvas]",
				function(t) {
					t.preventDefault();
					var n = e.$(this);
					if (!n.data("offcanvasTrigger")) {
						e.offcanvasTrigger(n, e.Utils.options(n.attr("data-uk-offcanvas")));
						n.trigger("click")
					}
				}),
				n.on("keydown.uk.offcanvas",
				function(e) {
					27 === e.keyCode && i.hide()
				})
			},
			init: function() {
				var t = this;
				this.options = e.$.extend({
					target: !!t.element.is("a") && t.element.attr("href"),
					mode: "push"
				},
				this.options),
				this.on("click",
				function(e) {
					e.preventDefault(),
					i.show(t.options.target, t.options)
				})
			}
		}),
		e.offcanvas = i
	} (UIkit),
	function(e) {
		"use strict";
		function t(t, n, i) {
			var r, o = e.$.Deferred(),
			a = t,
			s = t;
			return i[0] === n[0] ? (o.resolve(), o.promise()) : ("object" == typeof t && (a = t[0], s = t[1] || t[0]), e.$body.css("overflow-x", "hidden"), r = function() {
				n && n.hide().removeClass("uk-active " + s + " uk-animation-reverse"),
				i.addClass(a).one(e.support.animation.end,
				function() {
					setTimeout(function() {
						i.removeClass("" + a).css({
							opacity: "",
							display: ""
						})
					},
					0),
					o.resolve(),
					e.$body.css("overflow-x", ""),
					n && n.css({
						opacity: "",
						display: ""
					})
				}.bind(this)).show()
			},
			i.css("animation-duration", this.options.duration + "ms"), n && n.length ? (n.css("animation-duration", this.options.duration + "ms"), n.css("display", "none").addClass(s + " uk-animation-reverse").one(e.support.animation.end,
			function() {
				r()
			}.bind(this)).css("display", "")) : (i.addClass("uk-active"), r()), o.promise())
		}
		var n;
		e.component("switcher", {
			defaults: {
				connect: !1,
				toggle: ">*",
				active: 0,
				animation: !1,
				duration: 200,
				swiping: !0
			},
			animating: !1,
			boot: function() {
				e.ready(function(t) {
					e.$("[data-uk-switcher]", t).each(function() {
						var t = e.$(this);
						if (!t.data("switcher")) e.switcher(t, e.Utils.options(t.attr("data-uk-switcher")))
					})
				})
			},
			init: function() {
				var t = this;
				this.on("click.uk.switcher", this.options.toggle,
				function(e) {
					e.preventDefault(),
					t.show(this)
				}),
				this.options.connect && (this.connect = e.$(this.options.connect), this.connect.length && (this.connect.on("click.uk.switcher", "[data-uk-switcher-item]",
				function(n) {
					n.preventDefault();
					var i = e.$(this).attr("data-uk-switcher-item");
					if (t.index != i) switch (i) {
					case "next":
					case "previous":
						t.show(t.index + ("next" == i ? 1 : -1));
						break;
					default:
						t.show(parseInt(i, 10))
					}
				}), this.options.swiping && this.connect.on("swipeRight swipeLeft",
				function(e) {
					e.preventDefault(),
					window.getSelection().toString() || t.show(t.index + ("swipeLeft" == e.type ? 1 : -1))
				}), this.update()))
			},
			update: function() {
				this.connect.children().removeClass("uk-active").attr("aria-hidden", "true");
				var e = this.find(this.options.toggle),
				t = e.filter(".uk-active");
				if (t.length) this.show(t, !1);
				else {
					if (!1 === this.options.active) return;
					t = e.eq(this.options.active),
					this.show(t.length ? t: e.eq(0), !1)
				}
				e.not(t).attr("aria-expanded", "false"),
				t.attr("aria-expanded", "true")
			},
			show: function(i, r) {
				if (!this.animating) {
					var o = this.find(this.options.toggle);
					isNaN(i) ? i = e.$(i) : (i = i < 0 ? o.length - 1 : i, i = o.eq(o[i] ? i: 0));
					var a = this,
					s = e.$(i),
					l = n[this.options.animation] ||
					function(e, i) {
						if (!a.options.animation) return n.none.apply(a);
						var r = a.options.animation.split(",");
						return 1 == r.length && (r[1] = r[0]),
						r[0] = r[0].trim(),
						r[1] = r[1].trim(),
						t.apply(a, [r, e, i])
					}; ! 1 !== r && e.support.animation || (l = n.none),
					s.hasClass("uk-disabled") || (o.attr("aria-expanded", "false"), s.attr("aria-expanded", "true"), o.filter(".uk-active").removeClass("uk-active"), s.addClass("uk-active"), this.options.connect && this.connect.length && (this.index = this.find(this.options.toggle).index(s), -1 == this.index && (this.index = 0), this.connect.each(function() {
						var t = e.$(this),
						n = e.$(t.children()),
						i = e.$(n.filter(".uk-active")),
						r = e.$(n.eq(a.index));
						a.animating = !0,
						l.apply(a, [i, r]).then(function() {
							i.removeClass("uk-active"),
							r.addClass("uk-active"),
							i.attr("aria-hidden", "true"),
							r.attr("aria-hidden", "false"),
							e.Utils.checkDisplay(r, !0),
							a.animating = !1
						})
					})), this.trigger("show.uk.switcher", [s]))
				}
			}
		}),
		n = {
			none: function() {
				var t = e.$.Deferred();
				return t.resolve(),
				t.promise()
			},
			fade: function(e, n) {
				return t.apply(this, ["uk-animation-fade", e, n])
			},
			"slide-bottom": function(e, n) {
				return t.apply(this, ["uk-animation-slide-bottom", e, n])
			},
			"slide-top": function(e, n) {
				return t.apply(this, ["uk-animation-slide-top", e, n])
			},
			"slide-vertical": function(e, n, i) {
				var r = ["uk-animation-slide-top", "uk-animation-slide-bottom"];
				return e && e.index() > n.index() && r.reverse(),
				t.apply(this, [r, e, n])
			},
			"slide-left": function(e, n) {
				return t.apply(this, ["uk-animation-slide-left", e, n])
			},
			"slide-right": function(e, n) {
				return t.apply(this, ["uk-animation-slide-right", e, n])
			},
			"slide-horizontal": function(e, n, i) {
				var r = ["uk-animation-slide-right", "uk-animation-slide-left"];
				return e && e.index() > n.index() && r.reverse(),
				t.apply(this, [r, e, n])
			},
			scale: function(e, n) {
				return t.apply(this, ["uk-animation-scale-up", e, n])
			}
		},
		e.switcher.animations = n
	} (UIkit),
	function(e) {
		"use strict";
		e.component("tab", {
			defaults: {
				target: ">li:not(.uk-tab-responsive, .uk-disabled)",
				connect: !1,
				active: 0,
				animation: !1,
				duration: 200,
				swiping: !0
			},
			boot: function() {
				e.ready(function(t) {
					e.$("[data-uk-tab]", t).each(function() {
						var t = e.$(this);
						if (!t.data("tab")) e.tab(t, e.Utils.options(t.attr("data-uk-tab")))
					})
				})
			},
			init: function() {
				var t = this;
				this.current = !1,
				this.on("click.uk.tab", this.options.target,
				function(n) {
					if (n.preventDefault(), !t.switcher || !t.switcher.animating) {
						var i = t.find(t.options.target).not(this);
						i.removeClass("uk-active").blur(),
						t.trigger("change.uk.tab", [e.$(this).addClass("uk-active"), t.current]),
						t.current = e.$(this),
						t.options.connect || (i.attr("aria-expanded", "false"), e.$(this).attr("aria-expanded", "true"))
					}
				}),
				this.options.connect && (this.connect = e.$(this.options.connect)),
				this.responsivetab = e.$('<li class="uk-tab-responsive uk-active"><a></a></li>').append('<div class="uk-dropdown uk-dropdown-small"><ul class="uk-nav uk-nav-dropdown"></ul><div>'),
				this.responsivetab.dropdown = this.responsivetab.find(".uk-dropdown"),
				this.responsivetab.lst = this.responsivetab.dropdown.find("ul"),
				this.responsivetab.caption = this.responsivetab.find("a:first"),
				this.element.hasClass("uk-tab-bottom") && this.responsivetab.dropdown.addClass("uk-dropdown-up"),
				this.responsivetab.lst.on("click.uk.tab", "a",
				function(n) {
					n.preventDefault(),
					n.stopPropagation();
					var i = e.$(this);
					t.element.children("li:not(.uk-tab-responsive)").eq(i.data("index")).trigger("click")
				}),
				this.on("show.uk.switcher change.uk.tab",
				function(e, n) {
					t.responsivetab.caption.html(n.text())
				}),
				this.element.append(this.responsivetab),
				this.options.connect && (this.switcher = e.switcher(this.element, {
					toggle: ">li:not(.uk-tab-responsive)",
					connect: this.options.connect,
					active: this.options.active,
					animation: this.options.animation,
					duration: this.options.duration,
					swiping: this.options.swiping
				})),
				e.dropdown(this.responsivetab, {
					mode: "click",
					preventflip: "y"
				}),
				t.trigger("change.uk.tab", [this.element.find(this.options.target).not(".uk-tab-responsive").filter(".uk-active")]),
				this.check(),
				e.$win.on("resize orientationchange", e.Utils.debounce(function() {
					t.element.is(":visible") && t.check()
				},
				100)),
				this.on("display.uk.check",
				function() {
					t.element.is(":visible") && t.check()
				})
			},
			check: function() {
				var t = this.element.children("li:not(.uk-tab-responsive)").removeClass("uk-hidden");
				if (t.length) {
					var n, i, r = t.eq(0).offset().top + Math.ceil(t.eq(0).height() / 2),
					o = !1;
					if (this.responsivetab.lst.empty(), t.each(function() {
						e.$(this).offset().top > r && (o = !0)
					}), o) for (var a = 0; a < t.length; a++)(n = e.$(t.eq(a))).find("a"),
					"none" == n.css("float") || n.attr("uk-dropdown") || (n.hasClass("uk-disabled") || ((i = e.$(n[0].outerHTML)).find("a").data("index", a), this.responsivetab.lst.append(i)), n.addClass("uk-hidden"));
					this.responsivetab[this.responsivetab.lst.children("li").length ? "removeClass": "addClass"]("uk-hidden")
				} else this.responsivetab.addClass("uk-hidden")
			}
		})
	} (UIkit),
	function(e) {
		"use strict";
		e.component("cover", {
			defaults: {
				automute: !0
			},
			boot: function() {
				e.ready(function(t) {
					e.$("[data-uk-cover]", t).each(function() {
						var t = e.$(this);
						if (!t.data("cover")) e.cover(t, e.Utils.options(t.attr("data-uk-cover")))
					})
				})
			},
			init: function() {
				if (this.parent = this.element.parent(), e.$win.on("load resize orientationchange", e.Utils.debounce(function() {
					this.check()
				}.bind(this), 100)), this.on("display.uk.check",
				function(e) {
					this.element.is(":visible") && this.check()
				}.bind(this)), this.check(), this.element.is("iframe") && this.options.automute) {
					var t = this.element.attr("src");
					this.element.attr("src", "").on("load",
					function() {
						this.contentWindow.postMessage('{ "event": "command", "func": "mute", "method":"setVolume", "value":0}', "*")
					}).attr("src", [t, t.indexOf("?") > -1 ? "&": "?", "enablejsapi=1&api=1"].join(""))
				}
			},
			check: function() {
				this.element.css({
					width: "",
					height: ""
				}),
				this.dimension = {
					w: this.element.width(),
					h: this.element.height()
				},
				this.element.attr("width") && !isNaN(this.element.attr("width")) && (this.dimension.w = this.element.attr("width")),
				this.element.attr("height") && !isNaN(this.element.attr("height")) && (this.dimension.h = this.element.attr("height")),
				this.ratio = this.dimension.w / this.dimension.h;
				var e, t, n = this.parent.width(),
				i = this.parent.height();
				n / this.ratio < i ? (e = Math.ceil(i * this.ratio), t = i) : (e = n, t = Math.ceil(n / this.ratio)),
				this.element.css({
					width: e,
					height: t
				})
			}
		})
	} (UIkit),
	function(e) {
		var t;
		window.UIkit && (t = e(UIkit)),
		"function" == typeof define && define.amd && define("uikit-tooltip", ["uikit"],
		function() {
			return t || e(UIkit)
		})
	} (function(e) {
		"use strict";
		var t, n, i;
		return e.component("tooltip", {
			defaults: {
				offset: 5,
				pos: "top",
				animation: !1,
				delay: 0,
				cls: "",
				activeClass: "uk-active",
				src: function(e) {
					var t = e.attr("title");
					return void 0 !== t && e.data("cached-title", t).removeAttr("title"),
					e.data("cached-title")
				}
			},
			tip: "",
			boot: function() {
				e.$html.on("mouseenter.tooltip.uikit focus.tooltip.uikit", "[data-uk-tooltip]",
				function(t) {
					var n = e.$(this);
					n.data("tooltip") || (e.tooltip(n, e.Utils.options(n.attr("data-uk-tooltip"))), n.trigger("mouseenter"))
				})
			},
			init: function() {
				var n = this;
				t || (t = e.$('<div class="uk-tooltip"></div>').appendTo("body")),
				this.on({
					focus: function(e) {
						n.show()
					},
					blur: function(e) {
						n.hide()
					},
					mouseenter: function(e) {
						n.show()
					},
					mouseleave: function(e) {
						n.hide()
					}
				})
			},
			show: function() {
				if (this.tip = "function" == typeof this.options.src ? this.options.src(this.element) : this.options.src, n && clearTimeout(n), i && clearInterval(i), "string" == typeof this.tip && this.tip.length) {
					t.stop().css({
						top: -2e3,
						visibility: "hidden"
					}).removeClass(this.options.activeClass).show(),
					t.html('<div class="uk-tooltip-inner">' + this.tip + "</div>");
					var r = this,
					o = e.$.extend({},
					this.element.offset(), {
						width: this.element[0].offsetWidth,
						height: this.element[0].offsetHeight
					}),
					a = t[0].offsetWidth,
					s = t[0].offsetHeight,
					l = "function" == typeof this.options.offset ? this.options.offset.call(this.element) : this.options.offset,
					c = "function" == typeof this.options.pos ? this.options.pos.call(this.element) : this.options.pos,
					u = c.split("-"),
					d = {
						display: "none",
						visibility: "visible",
						top: o.top + o.height + s,
						left: o.left
					};
					if ("fixed" == e.$html.css("position") || "fixed" == e.$body.css("position")) {
						var h = e.$("body").offset(),
						f = e.$("html").offset(),
						p = {
							top: f.top + h.top,
							left: f.left + h.left
						};
						o.left -= p.left,
						o.top -= p.top
					}
					"left" != u[0] && "right" != u[0] || "right" != e.langdirection || (u[0] = "left" == u[0] ? "right": "left");
					var m = {
						bottom: {
							top: o.top + o.height + l,
							left: o.left + o.width / 2 - a / 2
						},
						top: {
							top: o.top - s - l,
							left: o.left + o.width / 2 - a / 2
						},
						left: {
							top: o.top + o.height / 2 - s / 2,
							left: o.left - a - l
						},
						right: {
							top: o.top + o.height / 2 - s / 2,
							left: o.left + o.width + l
						}
					};
					e.$.extend(d, m[u[0]]),
					2 == u.length && (d.left = "left" == u[1] ? o.left: o.left + o.width - a);
					var g = this.checkBoundary(d.left, d.top, a, s);
					if (g) {
						switch (g) {
						case "x":
							c = 2 == u.length ? u[0] + "-" + (d.left < 0 ? "left": "right") : d.left < 0 ? "right": "left";
							break;
						case "y":
							c = 2 == u.length ? (d.top < 0 ? "bottom": "top") + "-" + u[1] : d.top < 0 ? "bottom": "top";
							break;
						case "xy":
							c = 2 == u.length ? (d.top < 0 ? "bottom": "top") + "-" + (d.left < 0 ? "left": "right") : d.left < 0 ? "right": "left"
						}
						u = c.split("-"),
						e.$.extend(d, m[u[0]]),
						2 == u.length && (d.left = "left" == u[1] ? o.left: o.left + o.width - a)
					}
					d.left -= e.$body.position().left,
					n = setTimeout(function() {
						t.css(d).attr("class", ["uk-tooltip", "uk-tooltip-" + c, r.options.cls].join(" ")),
						r.options.animation ? t.css({
							opacity: 0,
							display: "block"
						}).addClass(r.options.activeClass).animate({
							opacity: 1
						},
						parseInt(r.options.animation, 10) || 400) : t.show().addClass(r.options.activeClass),
						n = !1,
						i = setInterval(function() {
							r.element.is(":visible") || r.hide()
						},
						150)
					},
					parseInt(this.options.delay, 10) || 0)
				}
			},
			hide: function() {
				if (!this.element.is("input") || this.element[0] !== document.activeElement) if (n && clearTimeout(n), i && clearInterval(i), t.stop(), this.options.animation) {
					var e = this;
					t.fadeOut(parseInt(this.options.animation, 10) || 400,
					function() {
						t.removeClass(e.options.activeClass)
					})
				} else t.hide().removeClass(this.options.activeClass)
			},
			content: function() {
				return this.tip
			},
			checkBoundary: function(t, n, i, r) {
				var o = "";
				return (t < 0 || t - e.$win.scrollLeft() + i > window.innerWidth) && (o += "x"),
				(n < 0 || n - e.$win.scrollTop() + r > window.innerHeight) && (o += "y"),
				o
			}
		}),
		e.tooltip
	}),
	function(e) {
		var t;
		window.UIkit && (t = e(UIkit)),
		"function" == typeof define && define.amd && define("uikit-slideshow", ["uikit"],
		function() {
			return t || e(UIkit)
		})
	} (function(e) {
		"use strict";
		var t, n = 0;
		e.component("slideshow", {
			defaults: {
				animation: "fade",
				duration: 500,
				height: "auto",
				start: 0,
				autoplay: !1,
				autoplayInterval: 7e3,
				videoautoplay: !0,
				videomute: !0,
				slices: 15,
				pauseOnHover: !0,
				kenburns: !1,
				kenburnsanimations: ["uk-animation-middle-left", "uk-animation-top-right", "uk-animation-bottom-left", "uk-animation-top-center", "", "uk-animation-bottom-right"]
			},
			current: !1,
			interval: null,
			hovering: !1,
			boot: function() {
				e.ready(function(t) {
					e.$("[data-uk-slideshow]", t).each(function() {
						var t = e.$(this);
						t.data("slideshow") || e.slideshow(t, e.Utils.options(t.attr("data-uk-slideshow")))
					})
				})
			},
			init: function() {
				var t = this;
				this.container = this.element.hasClass("uk-slideshow") ? this.element: e.$(this.find(".uk-slideshow:first")),
				this.current = this.options.start,
				this.animating = !1,
				this.fixFullscreen = navigator.userAgent.match(/(iPad|iPhone|iPod)/g) && this.container.hasClass("uk-slideshow-fullscreen"),
				this.options.kenburns && (this.kbanimduration = !0 === this.options.kenburns ? "15s": this.options.kenburns, String(this.kbanimduration).match(/(ms|s)$/) || (this.kbanimduration += "ms"), "string" == typeof this.options.kenburnsanimations && (this.options.kenburnsanimations = this.options.kenburnsanimations.split(","))),
				this.update(),
				this.on("click.uk.slideshow", "[data-uk-slideshow-item]",
				function(n) {
					n.preventDefault();
					var i = e.$(this).attr("data-uk-slideshow-item");
					if (t.current != i) {
						switch (i) {
						case "next":
						case "previous":
							t["next" == i ? "next": "previous"]();
							break;
						default:
							t.show(parseInt(i, 10))
						}
						t.stop()
					}
				}),
				e.$win.on("resize load", e.Utils.debounce(function() {
					t.resize(),
					t.fixFullscreen && (t.container.css("height", window.innerHeight), t.slides.css("height", window.innerHeight))
				},
				100)),
				setTimeout(function() {
					t.resize()
				},
				80),
				this.options.autoplay && this.start(),
				this.options.videoautoplay && this.slides.eq(this.current).data("media") && this.playmedia(this.slides.eq(this.current).data("media")),
				this.options.kenburns && this.applyKenBurns(this.slides.eq(this.current)),
				this.container.on({
					mouseenter: function() {
						t.options.pauseOnHover && (t.hovering = !0)
					},
					mouseleave: function() {
						t.hovering = !1
					}
				}),
				this.on("swipeRight swipeLeft",
				function(e) {
					t["swipeLeft" == e.type ? "next": "previous"]()
				}),
				this.on("display.uk.check",
				function() {
					t.element.is(":visible") && (t.resize(), t.fixFullscreen && (t.container.css("height", window.innerHeight), t.slides.css("height", window.innerHeight)))
				}),
				e.domObserve(this.element,
				function(e) {
					t.container.children(":not([data-slideshow-slide])").not(".uk-slideshow-ghost").length && t.update(!0)
				})
			},
			update: function(t) {
				var i, r = this,
				o = 0;
				this.slides = this.container.children(),
				this.slidesCount = this.slides.length,
				this.slides.eq(this.current).length || (this.current = 0),
				this.slides.each(function(t) {
					var a = e.$(this);
					if (!a.data("processed")) {
						var s = a.children("img,video,iframe").eq(0),
						l = "html";
						if (a.data("media", s), a.data("sizer", s), s.length) {
							var c;
							switch (l = s[0].nodeName.toLowerCase(), s[0].nodeName) {
							case "IMG":
								var u = e.$('<div class="uk-cover-background uk-position-cover"></div>').css({
									"background-image": "url(" + s.attr("src") + ")"
								});
								s.attr("width") && s.attr("height") && (c = e.$("<canvas></canvas>").attr({
									width: s.attr("width"),
									height: s.attr("height")
								}), s.replaceWith(c), s = c, c = void 0),
								s.css({
									width: "100%",
									height: "auto",
									opacity: 0
								}),
								a.prepend(u).data("cover", u);
								break;
							case "IFRAME":
								var d = s[0].src,
								h = "sw-" + ++n;
								s.attr("src", "").on("load",
								function() {
									if ((t !== r.current || t == r.current && !r.options.videoautoplay) && r.pausemedia(s), r.options.videomute) {
										r.mutemedia(s);
										var e = setInterval(function(t) {
											return function() {
												r.mutemedia(s),
												++t >= 4 && clearInterval(e)
											}
										} (0), 250)
									}
								}).data("slideshow", r).attr("data-player-id", h).attr("src", [d, d.indexOf("?") > -1 ? "&": "?", "enablejsapi=1&api=1&player_id=" + h].join("")).addClass("uk-position-absolute"),
								e.support.touch || s.css("pointer-events", "none"),
								c = !0,
								e.cover && (e.cover(s), s.attr("data-uk-cover", "{}"));
								break;
							case "VIDEO":
								s.addClass("uk-cover-object uk-position-absolute"),
								c = !0,
								r.options.videomute && r.mutemedia(s)
							}
							if (c) {
								i = e.$("<canvas></canvas>").attr({
									width: s[0].width,
									height: s[0].height
								});
								var f = e.$('<img style="width:100%;height:auto;">').attr("src", i[0].toDataURL());
								a.prepend(f),
								a.data("sizer", f)
							}
						} else a.data("sizer", a);
						r.hasKenBurns(a) && a.data("cover").css({
							"-webkit-animation-duration": r.kbanimduration,
							"animation-duration": r.kbanimduration
						}),
						a.data("processed", ++o),
						a.attr("data-slideshow-slide", l)
					}
				}),
				o && (this.triggers = this.find("[data-uk-slideshow-item]"), this.slides.attr("aria-hidden", "true").removeClass("uk-active").eq(this.current).addClass("uk-active").attr("aria-hidden", "false"), this.triggers.filter('[data-uk-slideshow-item="' + this.current + '"]').addClass("uk-active")),
				t && o && this.resize()
			},
			resize: function() {
				if (!this.container.hasClass("uk-slideshow-fullscreen")) {
					var t = this.options.height;
					"auto" === this.options.height && (t = 0, this.slides.css("height", "").each(function() {
						t = Math.max(t, e.$(this).height())
					})),
					this.container.css("height", t),
					this.slides.css("height", t)
				}
			},
			show: function(n, i) {
				if (!this.animating && this.current != n) {
					this.animating = !0;
					var r = this,
					o = this.slides.eq(this.current),
					a = this.slides.eq(n),
					s = i || (this.current < n ? 1 : -1),
					l = o.data("media"),
					c = t[this.options.animation] ? this.options.animation: "fade",
					u = a.data("media");
					r.applyKenBurns(a),
					e.support.animation || (c = "none"),
					o = e.$(o),
					a = e.$(a),
					r.trigger("beforeshow.uk.slideshow", [a, o, r]),
					t[c].apply(this, [o, a, s]).then(function() {
						r.animating && (l && l.is("video,iframe") && r.pausemedia(l), u && u.is("video,iframe") && r.playmedia(u), a.addClass("uk-active").attr("aria-hidden", "false"), o.removeClass("uk-active").attr("aria-hidden", "true"), r.animating = !1, r.current = n, e.Utils.checkDisplay(a, '[class*="uk-animation-"]:not(.uk-cover-background.uk-position-cover)'), r.trigger("show.uk.slideshow", [a, o, r]))
					}),
					r.triggers.removeClass("uk-active"),
					r.triggers.filter('[data-uk-slideshow-item="' + n + '"]').addClass("uk-active")
				}
			},
			applyKenBurns: function(e) {
				if (this.hasKenBurns(e)) {
					var t = this.options.kenburnsanimations,
					n = this.kbindex || 0;
					e.data("cover").attr("class", "uk-cover-background uk-position-cover").width(),
					e.data("cover").addClass(["uk-animation-scale", "uk-animation-reverse", t[n].trim()].join(" ")),
					this.kbindex = t[n + 1] ? n + 1 : 0
				}
			},
			hasKenBurns: function(e) {
				return this.options.kenburns && e.data("cover")
			},
			next: function() {
				this.show(this.slides[this.current + 1] ? this.current + 1 : 0, 1)
			},
			previous: function() {
				this.show(this.slides[this.current - 1] ? this.current - 1 : this.slides.length - 1, -1)
			},
			start: function() {
				this.stop();
				var e = this;
				this.interval = setInterval(function() {
					e.hovering || e.next()
				},
				this.options.autoplayInterval)
			},
			stop: function() {
				this.interval && clearInterval(this.interval)
			},
			playmedia: function(e) {
				if (e && e[0]) switch (e[0].nodeName) {
				case "VIDEO":
					this.options.videomute || (e[0].muted = !1),
					e[0].play();
					break;
				case "IFRAME":
					this.options.videomute || e[0].contentWindow.postMessage('{ "event": "command", "func": "unmute", "method":"setVolume", "value":1}', "*"),
					e[0].contentWindow.postMessage('{ "event": "command", "func": "playVideo", "method":"play"}', "*")
				}
			},
			pausemedia: function(e) {
				switch (e[0].nodeName) {
				case "VIDEO":
					e[0].pause();
					break;
				case "IFRAME":
					e[0].contentWindow.postMessage('{ "event": "command", "func": "pauseVideo", "method":"pause"}', "*")
				}
			},
			mutemedia: function(e) {
				switch (e[0].nodeName) {
				case "VIDEO":
					e[0].muted = !0;
					break;
				case "IFRAME":
					e[0].contentWindow.postMessage('{ "event": "command", "func": "mute", "method":"setVolume", "value":0}', "*")
				}
			}
		}),
		t = {
			none: function() {
				var t = e.$.Deferred();
				return t.resolve(),
				t.promise()
			},
			scroll: function(t, n, i) {
				var r = e.$.Deferred();
				return t.css("animation-duration", this.options.duration + "ms"),
				n.css("animation-duration", this.options.duration + "ms"),
				n.css("opacity", 1).one(e.support.animation.end,
				function() {
					t.css("opacity", 0).removeClass( - 1 == i ? "uk-slideshow-scroll-backward-out": "uk-slideshow-scroll-forward-out"),
					n.removeClass( - 1 == i ? "uk-slideshow-scroll-backward-in": "uk-slideshow-scroll-forward-in"),
					r.resolve()
				}.bind(this)),
				t.addClass( - 1 == i ? "uk-slideshow-scroll-backward-out": "uk-slideshow-scroll-forward-out"),
				n.addClass( - 1 == i ? "uk-slideshow-scroll-backward-in": "uk-slideshow-scroll-forward-in"),
				n.width(),
				r.promise()
			},
			swipe: function(t, n, i) {
				var r = e.$.Deferred();
				return t.css("animation-duration", this.options.duration + "ms"),
				n.css("animation-duration", this.options.duration + "ms"),
				n.css("opacity", 1).one(e.support.animation.end,
				function() {
					t.css("opacity", 0).removeClass( - 1 === i ? "uk-slideshow-swipe-backward-out": "uk-slideshow-swipe-forward-out"),
					n.removeClass( - 1 === i ? "uk-slideshow-swipe-backward-in": "uk-slideshow-swipe-forward-in"),
					r.resolve()
				}.bind(this)),
				t.addClass( - 1 == i ? "uk-slideshow-swipe-backward-out": "uk-slideshow-swipe-forward-out"),
				n.addClass( - 1 == i ? "uk-slideshow-swipe-backward-in": "uk-slideshow-swipe-forward-in"),
				n.width(),
				r.promise()
			},
			scale: function(t, n, i) {
				var r = e.$.Deferred();
				return t.css("animation-duration", this.options.duration + "ms"),
				n.css("animation-duration", this.options.duration + "ms"),
				n.css("opacity", 1),
				t.one(e.support.animation.end,
				function() {
					t.css("opacity", 0).removeClass("uk-slideshow-scale-out"),
					r.resolve()
				}.bind(this)),
				t.addClass("uk-slideshow-scale-out"),
				t.width(),
				r.promise()
			},
			fade: function(t, n, i) {
				var r = e.$.Deferred();
				return t.css("animation-duration", this.options.duration + "ms"),
				n.css("animation-duration", this.options.duration + "ms"),
				n.css("opacity", 1),
				n.data("cover") || n.data("placeholder") || n.css("opacity", 1).one(e.support.animation.end,
				function() {
					n.removeClass("uk-slideshow-fade-in")
				}).addClass("uk-slideshow-fade-in"),
				t.one(e.support.animation.end,
				function() {
					t.css("opacity", 0).removeClass("uk-slideshow-fade-out"),
					r.resolve()
				}.bind(this)),
				t.addClass("uk-slideshow-fade-out"),
				t.width(),
				r.promise()
			}
		},
		e.slideshow.animations = t,
		window.addEventListener("message",
		function(t) {
			var n, i = t.data;
			if ("string" == typeof i) try {
				i = JSON.parse(i)
			} catch(e) {
				i = {}
			}
			t.origin && t.origin.indexOf("vimeo") > -1 && "ready" == i.event && i.player_id && (n = e.$('[data-player-id="' + i.player_id + '"]')).length && n.data("slideshow").mutemedia(n)
		},
		!1)
	}),
	function(e) {
		var t;
		window.UIkit && (t = e(UIkit)),
		"function" == typeof define && define.amd && define("uikit-datepicker", ["uikit"],
		function() {
			return t || e(UIkit)
		})
	} (function(e) {
		"use strict";
		var t, n, i = !1;
		return e.component("datepicker", {
			defaults: {
				mobile: !1,
				weekstart: 1,
				i18n: {
					months: ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"],
					weekdays: ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
				},
				format: "YYYY-MM-DD",
				offsettop: 5,
				maxDate: !1,
				minDate: !1,
				pos: "auto",
				template: function(t, n) {
					var i, r = "";
					if (r += '<div class="uk-datepicker-nav">', r += '<a href="" class="uk-datepicker-previous"></a>', r += '<a href="" class="uk-datepicker-next"></a>', e.formSelect) {
						var o, a, s, l = (new Date).getFullYear(),
						c = [];
						for (i = 0; i < n.i18n.months.length; i++) i == t.month ? c.push('<option value="' + i + '" selected>' + n.i18n.months[i] + "</option>") : c.push('<option value="' + i + '">' + n.i18n.months[i] + "</option>");
						for (o = '<span class="uk-form-select">' + n.i18n.months[t.month] + '<select class="update-picker-month">' + c.join("") + "</select></span>", c = [], a = t.minDate ? t.minDate.year() : l - 50, s = t.maxDate ? t.maxDate.year() : l + 20, i = a; i <= s; i++) i == t.year ? c.push('<option value="' + i + '" selected>' + i + "</option>") : c.push('<option value="' + i + '">' + i + "</option>");
						r += '<div class="uk-datepicker-heading">' + o + " " + ('<span class="uk-form-select">' + t.year + '<select class="update-picker-year">' + c.join("") + "</select></span>") + "</div>"
					} else r += '<div class="uk-datepicker-heading">' + n.i18n.months[t.month] + " " + t.year + "</div>";
					for (r += "</div>", r += '<table class="uk-datepicker-table">', r += "<thead>", i = 0; i < t.weekdays.length; i++) t.weekdays[i] && (r += "<th>" + t.weekdays[i] + "</th>");
					for (r += "</thead>", r += "<tbody>", i = 0; i < t.days.length; i++) if (t.days[i] && t.days[i].length) {
						r += "<tr>";
						for (var u = 0; u < t.days[i].length; u++) if (t.days[i][u]) {
							var d = t.days[i][u],
							h = [];
							d.inmonth || h.push("uk-datepicker-table-muted"),
							d.selected && h.push("uk-active"),
							d.disabled && h.push("uk-datepicker-date-disabled uk-datepicker-table-muted"),
							r += '<td><a href="" class="' + h.join(" ") + '" data-date="' + d.day.format() + '">' + d.day.format("D") + "</a></td>"
						}
						r += "</tr>"
					}
					return r += "</tbody>",
					r += "</table>"
				}
			},
			boot: function() {
				e.$win.on("resize orientationchange",
				function() {
					i && i.hide()
				}),
				e.$html.on("focus.datepicker.uikit", "[data-uk-datepicker]",
				function(t) {
					var n = e.$(this);
					n.data("datepicker") || (t.preventDefault(), e.datepicker(n, e.Utils.options(n.attr("data-uk-datepicker"))), n.trigger("focus"))
				}),
				e.$html.on("click focus", "*",
				function(n) {
					var r = e.$(n.target); ! i || r[0] == t[0] || r.data("datepicker") || r.parents(".uk-datepicker:first").length || i.hide()
				})
			},
			init: function() {
				if (!e.support.touch || "date" != this.element.attr("type") || this.options.mobile) {
					var r = this;
					this.current = this.element.val() ? n(this.element.val(), this.options.format) : n(),
					this.on("click focus",
					function() {
						i !== r && r.pick(this.value ? this.value: "")
					}).on("change",
					function() {
						r.element.val() && !n(r.element.val(), r.options.format).isValid() && r.element.val(n().format(r.options.format))
					}),
					t || ((t = e.$('<div class="uk-dropdown uk-datepicker"></div>')).on("click", ".uk-datepicker-next, .uk-datepicker-previous, [data-date]",
					function(t) {
						t.stopPropagation(),
						t.preventDefault();
						var r = e.$(this);
						if (r.hasClass("uk-datepicker-date-disabled")) return ! 1;
						r.is("[data-date]") ? (i.current = n(r.data("date")), i.element.val(i.current.isValid() ? i.current.format(i.options.format) : null).trigger("change"), i.hide()) : i.add(r.hasClass("uk-datepicker-next") ? 1 : -1, "months")
					}), t.on("change", ".update-picker-month, .update-picker-year",
					function() {
						var t = e.$(this);
						i[t.is(".update-picker-year") ? "setYear": "setMonth"](Number(t.val()))
					}), t.appendTo("body"))
				}
			},
			pick: function(r) {
				var o = this.element.offset(),
				a = {
					left: o.left,
					right: ""
				};
				this.current = isNaN(r) ? n(r, this.options.format) : n(),
				this.initdate = this.current.format("YYYY-MM-DD"),
				this.update(),
				"right" == e.langdirection && (a.right = window.innerWidth - (a.left + this.element.outerWidth()), a.left = "");
				var s = o.top - this.element.outerHeight() + this.element.height() - this.options.offsettop - t.outerHeight(),
				l = o.top + this.element.outerHeight() + this.options.offsettop;
				a.top = l,
				"top" == this.options.pos ? a.top = s: "auto" == this.options.pos && window.innerHeight - l - t.outerHeight() < 0 && s >= 0 && (a.top = s),
				t.css(a).show(),
				this.trigger("show.uk.datepicker"),
				i = this
			},
			add: function(e, t) {
				this.current.add(e, t),
				this.update()
			},
			setMonth: function(e) {
				this.current.month(e),
				this.update()
			},
			setYear: function(e) {
				this.current.year(e),
				this.update()
			},
			update: function() {
				var e = this.getRows(this.current.year(), this.current.month()),
				n = this.options.template(e, this.options);
				t.html(n),
				this.trigger("update.uk.datepicker")
			},
			getRows: function(e, t) {
				var i = this.options,
				r = n().format("YYYY-MM-DD"),
				o = [31, e % 4 == 0 && e % 100 != 0 || e % 400 == 0 ? 29 : 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][t],
				a = new Date(e, t, 1, 12).getDay(),
				s = {
					month: t,
					year: e,
					weekdays: [],
					days: [],
					maxDate: !1,
					minDate: !1
				},
				l = []; ! 1 !== i.maxDate && (s.maxDate = isNaN(i.maxDate) ? n(i.maxDate, i.format).startOf("day").hours(12) : n().add(i.maxDate, "days").startOf("day").hours(12)),
				!1 !== i.minDate && (s.minDate = isNaN(i.minDate) ? n(i.minDate, i.format).startOf("day").hours(12) : n().add(i.minDate - 1, "days").startOf("day").hours(12)),
				s.weekdays = function() {
					for (var e = 0,
					t = []; e < 7; e++) {
						for (var n = e + (i.weekstart || 0); n >= 7;) n -= 7;
						t.push(i.i18n.weekdays[n])
					}
					return t
				} (),
				i.weekstart && i.weekstart > 0 && (a -= i.weekstart) < 0 && (a += 7);
				for (var c = o + a,
				u = c; u > 7;) u -= 7;
				c += 7 - u;
				for (var d, h, f, p, m, g = 0,
				v = 0; g < c; g++) d = new Date(e, t, g - a + 1, 12),
				h = s.minDate && s.minDate > d || s.maxDate && d > s.maxDate,
				m = !(g < a || g >= o + a),
				d = n(d),
				f = this.initdate == d.format("YYYY-MM-DD"),
				p = r == d.format("YYYY-MM-DD"),
				l.push({
					selected: f,
					today: p,
					disabled: h,
					day: d,
					inmonth: m
				}),
				7 == ++v && (s.days.push(l), l = [], v = 0);
				return s
			},
			hide: function() {
				i && i === this && (t.hide(), i = !1, this.trigger("hide.uk.datepicker"))
			}
		}),
		n = function(e) {
			function t(e, t, n) {
				switch (arguments.length) {
				case 2:
					return null != e ? e: t;
				case 3:
					return null != e ? e: null != t ? t: n;
				default:
					throw new Error("Implement me")
				}
			}
			function n(e, t) {
				return ye.call(e, t)
			}
			function i() {
				return {
					empty: !1,
					unusedTokens: [],
					unusedInput: [],
					overflow: -2,
					charsLeftOver: 0,
					nullInput: !1,
					invalidMonth: null,
					invalidFormat: !1,
					userInvalidated: !1,
					iso: !1
				}
			}
			function r(e) { ! 1 === me.suppressDeprecationWarnings && "undefined" != typeof console && console.warn && console.warn("Deprecation warning: " + e)
			}
			function o(e, t) {
				var n = !0;
				return d(function() {
					return n && (r(e), n = !1),
					t.apply(this, arguments)
				},
				t)
			}
			function a(e, t) {
				lt[e] || (r(t), lt[e] = !0)
			}
			function s(e, t) {
				return function(n) {
					return p(e.call(this, n), t)
				}
			}
			function l() {}
			function c(e, t) { ! 1 !== t && L(e),
				h(this, e),
				this._d = new Date( + e._d)
			}
			function u(e) {
				var t = _(e),
				n = t.year || 0,
				i = t.quarter || 0,
				r = t.month || 0,
				o = t.week || 0,
				a = t.day || 0,
				s = t.hour || 0,
				l = t.minute || 0,
				c = t.second || 0,
				u = t.millisecond || 0;
				this._milliseconds = +u + 1e3 * c + 6e4 * l + 36e5 * s,
				this._days = +a + 7 * o,
				this._months = +r + 3 * i + 12 * n,
				this._data = {},
				this._locale = me.localeData(),
				this._bubble()
			}
			function d(e, t) {
				for (var i in t) n(t, i) && (e[i] = t[i]);
				return n(t, "toString") && (e.toString = t.toString),
				n(t, "valueOf") && (e.valueOf = t.valueOf),
				e
			}
			function h(e, t) {
				var n, i, r;
				if (void 0 !== t._isAMomentObject && (e._isAMomentObject = t._isAMomentObject), void 0 !== t._i && (e._i = t._i), void 0 !== t._f && (e._f = t._f), void 0 !== t._l && (e._l = t._l), void 0 !== t._strict && (e._strict = t._strict), void 0 !== t._tzm && (e._tzm = t._tzm), void 0 !== t._isUTC && (e._isUTC = t._isUTC), void 0 !== t._offset && (e._offset = t._offset), void 0 !== t._pf && (e._pf = t._pf), void 0 !== t._locale && (e._locale = t._locale), Te.length > 0) for (n in Te) void 0 !== (r = t[i = Te[n]]) && (e[i] = r);
				return e
			}
			function f(e) {
				return e < 0 ? Math.ceil(e) : Math.floor(e)
			}
			function p(e, t, n) {
				for (var i = "" + Math.abs(e), r = e >= 0; i.length < t;) i = "0" + i;
				return (r ? n ? "+": "": "-") + i
			}
			function m(e, t) {
				var n = {
					milliseconds: 0,
					months: 0
				};
				return n.months = t.month() - e.month() + 12 * (t.year() - e.year()),
				e.clone().add(n.months, "M").isAfter(t) && --n.months,
				n.milliseconds = +t - +e.clone().add(n.months, "M"),
				n
			}
			function g(e, t) {
				var n;
				return t = $(t, e),
				e.isBefore(t) ? n = m(e, t) : ((n = m(t, e)).milliseconds = -n.milliseconds, n.months = -n.months),
				n
			}
			function v(e, t) {
				return function(n, i) {
					var r, o;
					return null === i || isNaN( + i) || (a(t, "moment()." + t + "(period, number) is deprecated. Please use moment()." + t + "(number, period)."), o = n, n = i, i = o),
					n = "string" == typeof n ? +n: n,
					r = me.duration(n, i),
					y(this, r, e),
					this
				}
			}
			function y(e, t, n, i) {
				var r = t._milliseconds,
				o = t._days,
				a = t._months;
				i = null == i || i,
				r && e._d.setTime( + e._d + r * n),
				o && de(e, "Date", ue(e, "Date") + o * n),
				a && ce(e, ue(e, "Month") + a * n),
				i && me.updateOffset(e, o || a)
			}
			function b(e) {
				return "[object Array]" === Object.prototype.toString.call(e)
			}
			function w(e) {
				return "[object Date]" === Object.prototype.toString.call(e) || e instanceof Date
			}
			function k(e, t, n) {
				var i, r = Math.min(e.length, t.length),
				o = Math.abs(e.length - t.length),
				a = 0;
				for (i = 0; i < r; i++)(n && e[i] !== t[i] || !n && C(e[i]) !== C(t[i])) && a++;
				return a + o
			}
			function x(e) {
				if (e) {
					var t = e.toLowerCase().replace(/(.)s$/, "$1");
					e = tt[e] || nt[t] || t
				}
				return e
			}
			function _(e) {
				var t, i, r = {};
				for (i in e) n(e, i) && (t = x(i)) && (r[t] = e[i]);
				return r
			}
			function C(e) {
				var t = +e,
				n = 0;
				return 0 !== t && isFinite(t) && (n = t >= 0 ? Math.floor(t) : Math.ceil(t)),
				n
			}
			function S(e, t) {
				return new Date(Date.UTC(e, t + 1, 0)).getUTCDate()
			}
			function M(e, t, n) {
				return oe(me([e, 11, 31 + t - n]), t, n).week
			}
			function T(e) {
				return D(e) ? 366 : 365
			}
			function D(e) {
				return e % 4 == 0 && e % 100 != 0 || e % 400 == 0
			}
			function L(e) {
				var t;
				e._a && -2 === e._pf.overflow && (t = e._a[we] < 0 || e._a[we] > 11 ? we: e._a[ke] < 1 || e._a[ke] > S(e._a[be], e._a[we]) ? ke: e._a[xe] < 0 || e._a[xe] > 23 ? xe: e._a[_e] < 0 || e._a[_e] > 59 ? _e: e._a[Ce] < 0 || e._a[Ce] > 59 ? Ce: e._a[Se] < 0 || e._a[Se] > 999 ? Se: -1, e._pf._overflowDayOfYear && (t < be || t > ke) && (t = ke), e._pf.overflow = t)
			}
			function O(e) {
				return null == e._isValid && (e._isValid = !isNaN(e._d.getTime()) && e._pf.overflow < 0 && !e._pf.empty && !e._pf.invalidMonth && !e._pf.nullInput && !e._pf.invalidFormat && !e._pf.userInvalidated, e._strict && (e._isValid = e._isValid && 0 === e._pf.charsLeftOver && 0 === e._pf.unusedTokens.length)),
				e._isValid
			}
			function N(e) {
				return e ? e.toLowerCase().replace("_", "-") : e
			}
			function A(e) {
				for (var t, n, i, r, o = 0; o < e.length;) {
					for (t = (r = N(e[o]).split("-")).length, n = (n = N(e[o + 1])) ? n.split("-") : null; t > 0;) {
						if (i = E(r.slice(0, t).join("-"))) return i;
						if (n && n.length >= t && k(r, n, !0) >= t - 1) break;
						t--
					}
					o++
				}
				return null
			}
			function E(e) {
				var t = null;
				if (!Me[e] && De) try {
					t = me.locale(),
					require("./locale/" + e),
					me.locale(t)
				} catch(e) {}
				return Me[e]
			}
			function $(e, t) {
				return t._isUTC ? me(e).zone(t._offset || 0) : me(e).local()
			}
			function q(e) {
				return e.match(/\[[\s\S]/) ? e.replace(/^\[|\]$/g, "") : e.replace(/\\/g, "")
			}
			function j(e) {
				var t, n, i = e.match(Ae);
				for (t = 0, n = i.length; t < n; t++) st[i[t]] ? i[t] = st[i[t]] : i[t] = q(i[t]);
				return function(r) {
					var o = "";
					for (t = 0; t < n; t++) o += i[t] instanceof Function ? i[t].call(r, e) : i[t];
					return o
				}
			}
			function P(e, t) {
				return e.isValid() ? (t = I(t, e.localeData()), it[t] || (it[t] = j(t)), it[t](e)) : e.localeData().invalidDate()
			}
			function I(e, t) {
				var n = 5;
				for (Ee.lastIndex = 0; n >= 0 && Ee.test(e);) e = e.replace(Ee,
				function(e) {
					return t.longDateFormat(e) || e
				}),
				Ee.lastIndex = 0,
				n -= 1;
				return e
			}
			function z(e, t) {
				var n = t._strict;
				switch (e) {
				case "Q":
					return Re;
				case "DDDD":
					return Be;
				case "YYYY":
				case "GGGG":
				case "gggg":
					return n ? Ge: je;
				case "Y":
				case "G":
				case "g":
					return Ke;
				case "YYYYYY":
				case "YYYYY":
				case "GGGGG":
				case "ggggg":
					return n ? Ve: Pe;
				case "S":
					if (n) return Re;
				case "SS":
					if (n) return Ue;
				case "SSS":
					if (n) return Be;
				case "DDD":
					return qe;
				case "MMM":
				case "MMMM":
				case "dd":
				case "ddd":
				case "dddd":
					return ze;
				case "a":
				case "A":
					return t._locale._meridiemParse;
				case "X":
					return Fe;
				case "Z":
				case "ZZ":
					return We;
				case "T":
					return He;
				case "SSSS":
					return Ie;
				case "MM":
				case "DD":
				case "YY":
				case "GG":
				case "gg":
				case "HH":
				case "hh":
				case "mm":
				case "ss":
				case "ww":
				case "WW":
					return n ? Ue: $e;
				case "M":
				case "D":
				case "d":
				case "H":
				case "h":
				case "m":
				case "s":
				case "w":
				case "W":
				case "e":
				case "E":
					return $e;
				case "Do":
					return Ye;
				default:
					return new RegExp(V(G(e.replace("\\", ""))))
				}
			}
			function W(e) {
				var t = (e = e || "").match(We) || [],
				n = ((t[t.length - 1] || []) + "").match(Je) || ["-", 0, 0],
				i = 60 * n[1] + C(n[2]);
				return "+" === n[0] ? -i: i
			}
			function H(e, t, n) {
				var i, r = n._a;
				switch (e) {
				case "Q":
					null != t && (r[we] = 3 * (C(t) - 1));
					break;
				case "M":
				case "MM":
					null != t && (r[we] = C(t) - 1);
					break;
				case "MMM":
				case "MMMM":
					null != (i = n._locale.monthsParse(t)) ? r[we] = i: n._pf.invalidMonth = t;
					break;
				case "D":
				case "DD":
					null != t && (r[ke] = C(t));
					break;
				case "Do":
					null != t && (r[ke] = C(parseInt(t, 10)));
					break;
				case "DDD":
				case "DDDD":
					null != t && (n._dayOfYear = C(t));
					break;
				case "YY":
					r[be] = me.parseTwoDigitYear(t);
					break;
				case "YYYY":
				case "YYYYY":
				case "YYYYYY":
					r[be] = C(t);
					break;
				case "a":
				case "A":
					n._isPm = n._locale.isPM(t);
					break;
				case "H":
				case "HH":
				case "h":
				case "hh":
					r[xe] = C(t);
					break;
				case "m":
				case "mm":
					r[_e] = C(t);
					break;
				case "s":
				case "ss":
					r[Ce] = C(t);
					break;
				case "S":
				case "SS":
				case "SSS":
				case "SSSS":
					r[Se] = C(1e3 * ("0." + t));
					break;
				case "X":
					n._d = new Date(1e3 * parseFloat(t));
					break;
				case "Z":
				case "ZZ":
					n._useUTC = !0,
					n._tzm = W(t);
					break;
				case "dd":
				case "ddd":
				case "dddd":
					null != (i = n._locale.weekdaysParse(t)) ? (n._w = n._w || {},
					n._w.d = i) : n._pf.invalidWeekday = t;
					break;
				case "w":
				case "ww":
				case "W":
				case "WW":
				case "d":
				case "e":
				case "E":
					e = e.substr(0, 1);
				case "gggg":
				case "GGGG":
				case "GGGGG":
					e = e.substr(0, 2),
					t && (n._w = n._w || {},
					n._w[e] = C(t));
					break;
				case "gg":
				case "GG":
					n._w = n._w || {},
					n._w[e] = me.parseTwoDigitYear(t)
				}
			}
			function F(e) {
				var n, i, r, o, a, s, l;
				null != (n = e._w).GG || null != n.W || null != n.E ? (a = 1, s = 4, i = t(n.GG, e._a[be], oe(me(), 1, 4).year), r = t(n.W, 1), o = t(n.E, 1)) : (a = e._locale._week.dow, s = e._locale._week.doy, i = t(n.gg, e._a[be], oe(me(), a, s).year), r = t(n.w, 1), null != n.d ? (o = n.d) < a && ++r: o = null != n.e ? n.e + a: a),
				l = ae(i, r, o, s, a),
				e._a[be] = l.year,
				e._dayOfYear = l.dayOfYear
			}
			function Y(e) {
				var n, i, r, o, a = [];
				if (!e._d) {
					for (r = U(e), e._w && null == e._a[ke] && null == e._a[we] && F(e), e._dayOfYear && (o = t(e._a[be], r[be]), e._dayOfYear > T(o) && (e._pf._overflowDayOfYear = !0), i = te(o, 0, e._dayOfYear), e._a[we] = i.getUTCMonth(), e._a[ke] = i.getUTCDate()), n = 0; n < 3 && null == e._a[n]; ++n) e._a[n] = a[n] = r[n];
					for (; n < 7; n++) e._a[n] = a[n] = null == e._a[n] ? 2 === n ? 1 : 0 : e._a[n];
					e._d = (e._useUTC ? te: ee).apply(null, a),
					null != e._tzm && e._d.setUTCMinutes(e._d.getUTCMinutes() + e._tzm)
				}
			}
			function R(e) {
				var t;
				e._d || (t = _(e._i), e._a = [t.year, t.month, t.day, t.hour, t.minute, t.second, t.millisecond], Y(e))
			}
			function U(e) {
				var t = new Date;
				return e._useUTC ? [t.getUTCFullYear(), t.getUTCMonth(), t.getUTCDate()] : [t.getFullYear(), t.getMonth(), t.getDate()]
			}
			function B(e) {
				if (e._f !== me.ISO_8601) {
					e._a = [],
					e._pf.empty = !0;
					var t, n, i, r, o, a = "" + e._i,
					s = a.length,
					l = 0;
					for (i = I(e._f, e._locale).match(Ae) || [], t = 0; t < i.length; t++) r = i[t],
					(n = (a.match(z(r, e)) || [])[0]) && ((o = a.substr(0, a.indexOf(n))).length > 0 && e._pf.unusedInput.push(o), a = a.slice(a.indexOf(n) + n.length), l += n.length),
					st[r] ? (n ? e._pf.empty = !1 : e._pf.unusedTokens.push(r), H(r, n, e)) : e._strict && !n && e._pf.unusedTokens.push(r);
					e._pf.charsLeftOver = s - l,
					a.length > 0 && e._pf.unusedInput.push(a),
					e._isPm && e._a[xe] < 12 && (e._a[xe] += 12),
					!1 === e._isPm && 12 === e._a[xe] && (e._a[xe] = 0),
					Y(e),
					L(e)
				} else Z(e)
			}
			function G(e) {
				return e.replace(/\\(\[)|\\(\])|\[([^\]\[]*)\]|\\(.)/g,
				function(e, t, n, i, r) {
					return t || n || i || r
				})
			}
			function V(e) {
				return e.replace(/[-\/\\^$*+?.()|[\]{}]/g, "\\$&")
			}
			function K(e) {
				var t, n, r, o, a;
				if (0 === e._f.length) return e._pf.invalidFormat = !0,
				void(e._d = new Date(NaN));
				for (o = 0; o < e._f.length; o++) a = 0,
				t = h({},
				e),
				null != e._useUTC && (t._useUTC = e._useUTC),
				t._pf = i(),
				t._f = e._f[o],
				B(t),
				O(t) && (a += t._pf.charsLeftOver, a += 10 * t._pf.unusedTokens.length, t._pf.score = a, (null == r || a < r) && (r = a, n = t));
				d(e, n || t)
			}
			function Z(e) {
				var t, n, i = e._i,
				r = Ze.exec(i);
				if (r) {
					for (e._pf.iso = !0, t = 0, n = Xe.length; t < n; t++) if (Xe[t][1].exec(i)) {
						e._f = Xe[t][0] + (r[6] || " ");
						break
					}
					for (t = 0, n = Qe.length; t < n; t++) if (Qe[t][1].exec(i)) {
						e._f += Qe[t][0];
						break
					}
					i.match(We) && (e._f += "Z"),
					B(e)
				} else e._isValid = !1
			}
			function X(e) {
				Z(e),
				!1 === e._isValid && (delete e._isValid, me.createFromInputFallback(e))
			}
			function Q(e, t) {
				var n, i = [];
				for (n = 0; n < e.length; ++n) i.push(t(e[n], n));
				return i
			}
			function J(t) {
				var n, i = t._i;
				i === e ? t._d = new Date: w(i) ? t._d = new Date( + i) : null !== (n = Le.exec(i)) ? t._d = new Date( + n[1]) : "string" == typeof i ? X(t) : b(i) ? (t._a = Q(i.slice(0),
				function(e) {
					return parseInt(e, 10)
				}), Y(t)) : "object" == typeof i ? R(t) : "number" == typeof i ? t._d = new Date(i) : me.createFromInputFallback(t)
			}
			function ee(e, t, n, i, r, o, a) {
				var s = new Date(e, t, n, i, r, o, a);
				return e < 1970 && s.setFullYear(e),
				s
			}
			function te(e) {
				var t = new Date(Date.UTC.apply(null, arguments));
				return e < 1970 && t.setUTCFullYear(e),
				t
			}
			function ne(e, t) {
				if ("string" == typeof e) if (isNaN(e)) {
					if ("number" != typeof(e = t.weekdaysParse(e))) return null
				} else e = parseInt(e, 10);
				return e
			}
			function ie(e, t, n, i, r) {
				return r.relativeTime(t || 1, !!n, e, i)
			}
			function re(e, t, n) {
				var i = me.duration(e).abs(),
				r = ve(i.as("s")),
				o = ve(i.as("m")),
				a = ve(i.as("h")),
				s = ve(i.as("d")),
				l = ve(i.as("M")),
				c = ve(i.as("y")),
				u = r < rt.s && ["s", r] || 1 === o && ["m"] || o < rt.m && ["mm", o] || 1 === a && ["h"] || a < rt.h && ["hh", a] || 1 === s && ["d"] || s < rt.d && ["dd", s] || 1 === l && ["M"] || l < rt.M && ["MM", l] || 1 === c && ["y"] || ["yy", c];
				return u[2] = t,
				u[3] = +e > 0,
				u[4] = n,
				ie.apply({},
				u)
			}
			function oe(e, t, n) {
				var i, r = n - t,
				o = n - e.day();
				return o > r && (o -= 7),
				o < r - 7 && (o += 7),
				i = me(e).add(o, "d"),
				{
					week: Math.ceil(i.dayOfYear() / 7),
					year: i.year()
				}
			}
			function ae(e, t, n, i, r) {
				var o, a, s = te(e, 0, 1).getUTCDay();
				return s = 0 === s ? 7 : s,
				n = null != n ? n: r,
				o = r - s + (s > i ? 7 : 0) - (s < r ? 7 : 0),
				a = 7 * (t - 1) + (n - r) + o + 1,
				{
					year: a > 0 ? e: e - 1,
					dayOfYear: a > 0 ? a: T(e - 1) + a
				}
			}
			function se(t) {
				var n = t._i,
				i = t._f;
				return t._locale = t._locale || me.localeData(t._l),
				null === n || i === e && "" === n ? me.invalid({
					nullInput: !0
				}) : ("string" == typeof n && (t._i = n = t._locale.preparse(n)), me.isMoment(n) ? new c(n, !0) : (i ? b(i) ? K(t) : B(t) : J(t), new c(t)))
			}
			function le(e, t) {
				var n, i;
				if (1 === t.length && b(t[0]) && (t = t[0]), !t.length) return me();
				for (n = t[0], i = 1; i < t.length; ++i) t[i][e](n) && (n = t[i]);
				return n
			}
			function ce(e, t) {
				var n;
				return "string" == typeof t && "number" != typeof(t = e.localeData().monthsParse(t)) ? e: (n = Math.min(e.date(), S(e.year(), t)), e._d["set" + (e._isUTC ? "UTC": "") + "Month"](t, n), e)
			}
			function ue(e, t) {
				return e._d["get" + (e._isUTC ? "UTC": "") + t]()
			}
			function de(e, t, n) {
				return "Month" === t ? ce(e, n) : e._d["set" + (e._isUTC ? "UTC": "") + t](n)
			}
			function he(e, t) {
				return function(n) {
					return null != n ? (de(this, e, n), me.updateOffset(this, t), this) : ue(this, e)
				}
			}
			function fe(e) {
				return 400 * e / 146097
			}
			function pe(e) {
				return 146097 * e / 400
			}
			"undefined" != typeof global && global;
			for (var me, ge, ve = Math.round,
			ye = Object.prototype.hasOwnProperty,
			be = 0,
			we = 1,
			ke = 2,
			xe = 3,
			_e = 4,
			Ce = 5,
			Se = 6,
			Me = {},
			Te = [], De = "undefined" != typeof module && module.exports, Le = /^\/?Date\((\-?\d+)/i, Oe = /(\-)?(?:(\d*)\.)?(\d+)\:(\d+)(?:\:(\d+)\.?(\d{3})?)?/, Ne = /^(-)?P(?:(?:([0-9,.]*)Y)?(?:([0-9,.]*)M)?(?:([0-9,.]*)D)?(?:T(?:([0-9,.]*)H)?(?:([0-9,.]*)M)?(?:([0-9,.]*)S)?)?|([0-9,.]*)W)$/, Ae = /(\[[^\[]*\])|(\\)?(Mo|MM?M?M?|Do|DDDo|DD?D?D?|ddd?d?|do?|w[o|w]?|W[o|W]?|Q|YYYYYY|YYYYY|YYYY|YY|gg(ggg?)?|GG(GGG?)?|e|E|a|A|hh?|HH?|mm?|ss?|S{1,4}|X|zz?|ZZ?|.)/g, Ee = /(\[[^\[]*\])|(\\)?(LT|LL?L?L?|l{1,4})/g, $e = /\d\d?/, qe = /\d{1,3}/, je = /\d{1,4}/, Pe = /[+\-]?\d{1,6}/, Ie = /\d+/, ze = /[0-9]*['a-z\u00A0-\u05FF\u0700-\uD7FF\uF900-\uFDCF\uFDF0-\uFFEF]+|[\u0600-\u06FF\/]+(\s*?[\u0600-\u06FF]+){1,2}/i, We = /Z|[\+\-]\d\d:?\d\d/gi, He = /T/i, Fe = /[\+\-]?\d+(\.\d{1,3})?/, Ye = /\d{1,2}/, Re = /\d/, Ue = /\d\d/, Be = /\d{3}/, Ge = /\d{4}/, Ve = /[+-]?\d{6}/, Ke = /[+-]?\d+/, Ze = /^\s*(?:[+-]\d{6}|\d{4})-(?:(\d\d-\d\d)|(W\d\d$)|(W\d\d-\d)|(\d\d\d))((T| )(\d\d(:\d\d(:\d\d(\.\d+)?)?)?)?([\+\-]\d\d(?::?\d\d)?|\s*Z)?)?$/, Xe = [["YYYYYY-MM-DD", /[+-]\d{6}-\d{2}-\d{2}/], ["YYYY-MM-DD", /\d{4}-\d{2}-\d{2}/], ["GGGG-[W]WW-E", /\d{4}-W\d{2}-\d/], ["GGGG-[W]WW", /\d{4}-W\d{2}/], ["YYYY-DDD", /\d{4}-\d{3}/]], Qe = [["HH:mm:ss.SSSS", /(T| )\d\d:\d\d:\d\d\.\d+/], ["HH:mm:ss", /(T| )\d\d:\d\d:\d\d/], ["HH:mm", /(T| )\d\d:\d\d/], ["HH", /(T| )\d\d/]], Je = /([\+\-]|\d\d)/gi, et = ("Date|Hours|Minutes|Seconds|Milliseconds".split("|"), {
				Milliseconds: 1,
				Seconds: 1e3,
				Minutes: 6e4,
				Hours: 36e5,
				Days: 864e5,
				Months: 2592e6,
				Years: 31536e6
			}), tt = {
				ms: "millisecond",
				s: "second",
				m: "minute",
				h: "hour",
				d: "day",
				D: "date",
				w: "week",
				W: "isoWeek",
				M: "month",
				Q: "quarter",
				y: "year",
				DDD: "dayOfYear",
				e: "weekday",
				E: "isoWeekday",
				gg: "weekYear",
				GG: "isoWeekYear"
			},
			nt = {
				dayofyear: "dayOfYear",
				isoweekday: "isoWeekday",
				isoweek: "isoWeek",
				weekyear: "weekYear",
				isoweekyear: "isoWeekYear"
			},
			it = {},
			rt = {
				s: 45,
				m: 45,
				h: 22,
				d: 26,
				M: 11
			},
			ot = "DDD w W M D d".split(" "), at = "M D H h m s w W".split(" "), st = {
				M: function() {
					return this.month() + 1
				},
				MMM: function(e) {
					return this.localeData().monthsShort(this, e)
				},
				MMMM: function(e) {
					return this.localeData().months(this, e)
				},
				D: function() {
					return this.date()
				},
				DDD: function() {
					return this.dayOfYear()
				},
				d: function() {
					return this.day()
				},
				dd: function(e) {
					return this.localeData().weekdaysMin(this, e)
				},
				ddd: function(e) {
					return this.localeData().weekdaysShort(this, e)
				},
				dddd: function(e) {
					return this.localeData().weekdays(this, e)
				},
				w: function() {
					return this.week()
				},
				W: function() {
					return this.isoWeek()
				},
				YY: function() {
					return p(this.year() % 100, 2)
				},
				YYYY: function() {
					return p(this.year(), 4)
				},
				YYYYY: function() {
					return p(this.year(), 5)
				},
				YYYYYY: function() {
					var e = this.year();
					return (e >= 0 ? "+": "-") + p(Math.abs(e), 6)
				},
				gg: function() {
					return p(this.weekYear() % 100, 2)
				},
				gggg: function() {
					return p(this.weekYear(), 4)
				},
				ggggg: function() {
					return p(this.weekYear(), 5)
				},
				GG: function() {
					return p(this.isoWeekYear() % 100, 2)
				},
				GGGG: function() {
					return p(this.isoWeekYear(), 4)
				},
				GGGGG: function() {
					return p(this.isoWeekYear(), 5)
				},
				e: function() {
					return this.weekday()
				},
				E: function() {
					return this.isoWeekday()
				},
				a: function() {
					return this.localeData().meridiem(this.hours(), this.minutes(), !0)
				},
				A: function() {
					return this.localeData().meridiem(this.hours(), this.minutes(), !1)
				},
				H: function() {
					return this.hours()
				},
				h: function() {
					return this.hours() % 12 || 12
				},
				m: function() {
					return this.minutes()
				},
				s: function() {
					return this.seconds()
				},
				S: function() {
					return C(this.milliseconds() / 100)
				},
				SS: function() {
					return p(C(this.milliseconds() / 10), 2)
				},
				SSS: function() {
					return p(this.milliseconds(), 3)
				},
				SSSS: function() {
					return p(this.milliseconds(), 3)
				},
				Z: function() {
					var e = -this.zone(),
					t = "+";
					return e < 0 && (e = -e, t = "-"),
					t + p(C(e / 60), 2) + ":" + p(C(e) % 60, 2)
				},
				ZZ: function() {
					var e = -this.zone(),
					t = "+";
					return e < 0 && (e = -e, t = "-"),
					t + p(C(e / 60), 2) + p(C(e) % 60, 2)
				},
				z: function() {
					return this.zoneAbbr()
				},
				zz: function() {
					return this.zoneName()
				},
				X: function() {
					return this.unix()
				},
				Q: function() {
					return this.quarter()
				}
			},
			lt = {},
			ct = ["months", "monthsShort", "weekdays", "weekdaysShort", "weekdaysMin"]; ot.length;) ge = ot.pop(),
			st[ge + "o"] = function(e, t) {
				return function(n) {
					return this.localeData().ordinal(e.call(this, n), t)
				}
			} (st[ge], ge);
			for (; at.length;) ge = at.pop(),
			st[ge + ge] = s(st[ge], 2);
			for (st.DDDD = s(st.DDD, 3), d(l.prototype, {
				set: function(e) {
					var t, n;
					for (n in e)"function" == typeof(t = e[n]) ? this[n] = t: this["_" + n] = t
				},
				_months: "January_February_March_April_May_June_July_August_September_October_November_December".split("_"),
				months: function(e) {
					return this._months[e.month()]
				},
				_monthsShort: "Jan_Feb_Mar_Apr_May_Jun_Jul_Aug_Sep_Oct_Nov_Dec".split("_"),
				monthsShort: function(e) {
					return this._monthsShort[e.month()]
				},
				monthsParse: function(e) {
					var t, n, i;
					for (this._monthsParse || (this._monthsParse = []), t = 0; t < 12; t++) if (this._monthsParse[t] || (n = me.utc([2e3, t]), i = "^" + this.months(n, "") + "|^" + this.monthsShort(n, ""), this._monthsParse[t] = new RegExp(i.replace(".", ""), "i")), this._monthsParse[t].test(e)) return t
				},
				_weekdays: "Sunday_Monday_Tuesday_Wednesday_Thursday_Friday_Saturday".split("_"),
				weekdays: function(e) {
					return this._weekdays[e.day()]
				},
				_weekdaysShort: "Sun_Mon_Tue_Wed_Thu_Fri_Sat".split("_"),
				weekdaysShort: function(e) {
					return this._weekdaysShort[e.day()]
				},
				_weekdaysMin: "Su_Mo_Tu_We_Th_Fr_Sa".split("_"),
				weekdaysMin: function(e) {
					return this._weekdaysMin[e.day()]
				},
				weekdaysParse: function(e) {
					var t, n, i;
					for (this._weekdaysParse || (this._weekdaysParse = []), t = 0; t < 7; t++) if (this._weekdaysParse[t] || (n = me([2e3, 1]).day(t), i = "^" + this.weekdays(n, "") + "|^" + this.weekdaysShort(n, "") + "|^" + this.weekdaysMin(n, ""), this._weekdaysParse[t] = new RegExp(i.replace(".", ""), "i")), this._weekdaysParse[t].test(e)) return t
				},
				_longDateFormat: {
					LT: "h:mm A",
					L: "MM/DD/YYYY",
					LL: "MMMM D, YYYY",
					LLL: "MMMM D, YYYY LT",
					LLLL: "dddd, MMMM D, YYYY LT"
				},
				longDateFormat: function(e) {
					var t = this._longDateFormat[e];
					return ! t && this._longDateFormat[e.toUpperCase()] && (t = this._longDateFormat[e.toUpperCase()].replace(/MMMM|MM|DD|dddd/g,
					function(e) {
						return e.slice(1)
					}), this._longDateFormat[e] = t),
					t
				},
				isPM: function(e) {
					return "p" === (e + "").toLowerCase().charAt(0)
				},
				_meridiemParse: /[ap]\.?m?\.?/i,
				meridiem: function(e, t, n) {
					return e > 11 ? n ? "pm": "PM": n ? "am": "AM"
				},
				_calendar: {
					sameDay: "[Today at] LT",
					nextDay: "[Tomorrow at] LT",
					nextWeek: "dddd [at] LT",
					lastDay: "[Yesterday at] LT",
					lastWeek: "[Last] dddd [at] LT",
					sameElse: "L"
				},
				calendar: function(e, t) {
					var n = this._calendar[e];
					return "function" == typeof n ? n.apply(t) : n
				},
				_relativeTime: {
					future: "in %s",
					past: "%s ago",
					s: "a few seconds",
					m: "a minute",
					mm: "%d minutes",
					h: "an hour",
					hh: "%d hours",
					d: "a day",
					dd: "%d days",
					M: "a month",
					MM: "%d months",
					y: "a year",
					yy: "%d years"
				},
				relativeTime: function(e, t, n, i) {
					var r = this._relativeTime[n];
					return "function" == typeof r ? r(e, t, n, i) : r.replace(/%d/i, e)
				},
				pastFuture: function(e, t) {
					var n = this._relativeTime[e > 0 ? "future": "past"];
					return "function" == typeof n ? n(t) : n.replace(/%s/i, t)
				},
				ordinal: function(e) {
					return this._ordinal.replace("%d", e)
				},
				_ordinal: "%d",
				preparse: function(e) {
					return e
				},
				postformat: function(e) {
					return e
				},
				week: function(e) {
					return oe(e, this._week.dow, this._week.doy).week
				},
				_week: {
					dow: 0,
					doy: 6
				},
				_invalidDate: "Invalid date",
				invalidDate: function() {
					return this._invalidDate
				}
			}), (me = function(t, n, r, o) {
				var a;
				return "boolean" == typeof r && (o = r, r = e),
				a = {},
				a._isAMomentObject = !0,
				a._i = t,
				a._f = n,
				a._l = r,
				a._strict = o,
				a._isUTC = !1,
				a._pf = i(),
				se(a)
			}).suppressDeprecationWarnings = !1, me.createFromInputFallback = o("moment construction falls back to js Date. This is discouraged and will be removed in upcoming major release. Please refer to https://github.com/moment/moment/issues/1407 for more info.",
			function(e) {
				e._d = new Date(e._i)
			}), me.min = function() {
				return le("isBefore", [].slice.call(arguments, 0))
			},
			me.max = function() {
				return le("isAfter", [].slice.call(arguments, 0))
			},
			me.utc = function(t, n, r, o) {
				var a;
				return "boolean" == typeof r && (o = r, r = e),
				a = {},
				a._isAMomentObject = !0,
				a._useUTC = !0,
				a._isUTC = !0,
				a._l = r,
				a._i = t,
				a._f = n,
				a._strict = o,
				a._pf = i(),
				se(a).utc()
			},
			me.unix = function(e) {
				return me(1e3 * e)
			},
			me.duration = function(e, t) {
				var i, r, o, a, s = e,
				l = null;
				return me.isDuration(e) ? s = {
					ms: e._milliseconds,
					d: e._days,
					M: e._months
				}: "number" == typeof e ? (s = {},
				t ? s[t] = e: s.milliseconds = e) : (l = Oe.exec(e)) ? (i = "-" === l[1] ? -1 : 1, s = {
					y: 0,
					d: C(l[ke]) * i,
					h: C(l[xe]) * i,
					m: C(l[_e]) * i,
					s: C(l[Ce]) * i,
					ms: C(l[Se]) * i
				}) : (l = Ne.exec(e)) ? (i = "-" === l[1] ? -1 : 1, s = {
					y: (o = function(e) {
						var t = e && parseFloat(e.replace(",", "."));
						return (isNaN(t) ? 0 : t) * i
					})(l[2]),
					M: o(l[3]),
					d: o(l[4]),
					h: o(l[5]),
					m: o(l[6]),
					s: o(l[7]),
					w: o(l[8])
				}) : "object" == typeof s && ("from" in s || "to" in s) && (a = g(me(s.from), me(s.to)), (s = {}).ms = a.milliseconds, s.M = a.months),
				r = new u(s),
				me.isDuration(e) && n(e, "_locale") && (r._locale = e._locale),
				r
			},
			me.version = "2.8.3", me.defaultFormat = "YYYY-MM-DDTHH:mm:ssZ", me.ISO_8601 = function() {},
			me.momentProperties = Te, me.updateOffset = function() {},
			me.relativeTimeThreshold = function(t, n) {
				return rt[t] !== e && (n === e ? rt[t] : (rt[t] = n, !0))
			},
			me.lang = o("moment.lang is deprecated. Use moment.locale instead.",
			function(e, t) {
				return me.locale(e, t)
			}), me.locale = function(e, t) {
				var n;
				return e && (n = void 0 !== t ? me.defineLocale(e, t) : me.localeData(e)) && (me.duration._locale = me._locale = n),
				me._locale._abbr
			},
			me.defineLocale = function(e, t) {
				return null !== t ? (t.abbr = e, Me[e] || (Me[e] = new l), Me[e].set(t), me.locale(e), Me[e]) : (delete Me[e], null)
			},
			me.langData = o("moment.langData is deprecated. Use moment.localeData instead.",
			function(e) {
				return me.localeData(e)
			}), me.localeData = function(e) {
				var t;
				if (e && e._locale && e._locale._abbr && (e = e._locale._abbr), !e) return me._locale;
				if (!b(e)) {
					if (t = E(e)) return t;
					e = [e]
				}
				return A(e)
			},
			me.isMoment = function(e) {
				return e instanceof c || null != e && n(e, "_isAMomentObject")
			},
			me.isDuration = function(e) {
				return e instanceof u
			},
			ge = ct.length - 1; ge >= 0; --ge) !
			function(t) {
				var n, i;
				if (0 === t.indexOf("week")) n = 7,
				i = "day";
				else {
					if (0 !== t.indexOf("month")) return;
					n = 12,
					i = "month"
				}
				me[t] = function(r, o) {
					var a, s, l = me._locale[t],
					c = [];
					if ("number" == typeof r && (o = r, r = e), s = function(e) {
						var t = me().utc().set(i, e);
						return l.call(me._locale, t, r || "")
					},
					null != o) return s(o);
					for (a = 0; a < n; a++) c.push(s(a));
					return c
				}
			} (ct[ge]);
			me.normalizeUnits = function(e) {
				return x(e)
			},
			me.invalid = function(e) {
				var t = me.utc(NaN);
				return null != e ? d(t._pf, e) : t._pf.userInvalidated = !0,
				t
			},
			me.parseZone = function() {
				return me.apply(null, arguments).parseZone()
			},
			me.parseTwoDigitYear = function(e) {
				return C(e) + (C(e) > 68 ? 1900 : 2e3)
			},
			d(me.fn = c.prototype, {
				clone: function() {
					return me(this)
				},
				valueOf: function() {
					return + this._d + 6e4 * (this._offset || 0)
				},
				unix: function() {
					return Math.floor( + this / 1e3)
				},
				toString: function() {
					return this.clone().locale("en").format("ddd MMM DD YYYY HH:mm:ss [GMT]ZZ")
				},
				toDate: function() {
					return this._offset ? new Date( + this) : this._d
				},
				toISOString: function() {
					var e = me(this).utc();
					return 0 < e.year() && e.year() <= 9999 ? P(e, "YYYY-MM-DD[T]HH:mm:ss.SSS[Z]") : P(e, "YYYYYY-MM-DD[T]HH:mm:ss.SSS[Z]")
				},
				toArray: function() {
					var e = this;
					return [e.year(), e.month(), e.date(), e.hours(), e.minutes(), e.seconds(), e.milliseconds()]
				},
				isValid: function() {
					return O(this)
				},
				isDSTShifted: function() {
					return !! this._a && (this.isValid() && k(this._a, (this._isUTC ? me.utc(this._a) : me(this._a)).toArray()) > 0)
				},
				parsingFlags: function() {
					return d({},
					this._pf)
				},
				invalidAt: function() {
					return this._pf.overflow
				},
				utc: function(e) {
					return this.zone(0, e)
				},
				local: function(e) {
					return this._isUTC && (this.zone(0, e), this._isUTC = !1, e && this.add(this._dateTzOffset(), "m")),
					this
				},
				format: function(e) {
					var t = P(this, e || me.defaultFormat);
					return this.localeData().postformat(t)
				},
				add: v(1, "add"),
				subtract: v( - 1, "subtract"),
				diff: function(e, t, n) {
					var i, r, o, a = $(e, this),
					s = 6e4 * (this.zone() - a.zone());
					return "year" === (t = x(t)) || "month" === t ? (i = 432e5 * (this.daysInMonth() + a.daysInMonth()), r = 12 * (this.year() - a.year()) + (this.month() - a.month()), o = this - me(this).startOf("month") - (a - me(a).startOf("month")), r += (o -= 6e4 * (this.zone() - me(this).startOf("month").zone() - (a.zone() - me(a).startOf("month").zone()))) / i, "year" === t && (r /= 12)) : (i = this - a, r = "second" === t ? i / 1e3: "minute" === t ? i / 6e4: "hour" === t ? i / 36e5: "day" === t ? (i - s) / 864e5: "week" === t ? (i - s) / 6048e5: i),
					n ? r: f(r)
				},
				from: function(e, t) {
					return me.duration({
						to: this,
						from: e
					}).locale(this.locale()).humanize(!t)
				},
				fromNow: function(e) {
					return this.from(me(), e)
				},
				calendar: function(e) {
					var t = $(e || me(), this).startOf("day"),
					n = this.diff(t, "days", !0),
					i = n < -6 ? "sameElse": n < -1 ? "lastWeek": n < 0 ? "lastDay": n < 1 ? "sameDay": n < 2 ? "nextDay": n < 7 ? "nextWeek": "sameElse";
					return this.format(this.localeData().calendar(i, this))
				},
				isLeapYear: function() {
					return D(this.year())
				},
				isDST: function() {
					return this.zone() < this.clone().month(0).zone() || this.zone() < this.clone().month(5).zone()
				},
				day: function(e) {
					var t = this._isUTC ? this._d.getUTCDay() : this._d.getDay();
					return null != e ? (e = ne(e, this.localeData()), this.add(e - t, "d")) : t
				},
				month: he("Month", !0),
				startOf: function(e) {
					switch (e = x(e)) {
					case "year":
						this.month(0);
					case "quarter":
					case "month":
						this.date(1);
					case "week":
					case "isoWeek":
					case "day":
						this.hours(0);
					case "hour":
						this.minutes(0);
					case "minute":
						this.seconds(0);
					case "second":
						this.milliseconds(0)
					}
					return "week" === e ? this.weekday(0) : "isoWeek" === e && this.isoWeekday(1),
					"quarter" === e && this.month(3 * Math.floor(this.month() / 3)),
					this
				},
				endOf: function(e) {
					return e = x(e),
					this.startOf(e).add(1, "isoWeek" === e ? "week": e).subtract(1, "ms")
				},
				isAfter: function(e, t) {
					return "millisecond" === (t = x(void 0 !== t ? t: "millisecond")) ? (e = me.isMoment(e) ? e: me(e), +this > +e) : +this.clone().startOf(t) > +me(e).startOf(t)
				},
				isBefore: function(e, t) {
					return "millisecond" === (t = x(void 0 !== t ? t: "millisecond")) ? (e = me.isMoment(e) ? e: me(e), +this < +e) : +this.clone().startOf(t) < +me(e).startOf(t)
				},
				isSame: function(e, t) {
					return "millisecond" === (t = x(t || "millisecond")) ? (e = me.isMoment(e) ? e: me(e), +this == +e) : +this.clone().startOf(t) == +$(e, this).startOf(t)
				},
				min: o("moment().min is deprecated, use moment.min instead. https://github.com/moment/moment/issues/1548",
				function(e) {
					return (e = me.apply(null, arguments)) < this ? this: e
				}),
				max: o("moment().max is deprecated, use moment.max instead. https://github.com/moment/moment/issues/1548",
				function(e) {
					return (e = me.apply(null, arguments)) > this ? this: e
				}),
				zone: function(e, t) {
					var n, i = this._offset || 0;
					return null == e ? this._isUTC ? i: this._dateTzOffset() : ("string" == typeof e && (e = W(e)), Math.abs(e) < 16 && (e *= 60), !this._isUTC && t && (n = this._dateTzOffset()), this._offset = e, this._isUTC = !0, null != n && this.subtract(n, "m"), i !== e && (!t || this._changeInProgress ? y(this, me.duration(i - e, "m"), 1, !1) : this._changeInProgress || (this._changeInProgress = !0, me.updateOffset(this, !0), this._changeInProgress = null)), this)
				},
				zoneAbbr: function() {
					return this._isUTC ? "UTC": ""
				},
				zoneName: function() {
					return this._isUTC ? "Coordinated Universal Time": ""
				},
				parseZone: function() {
					return this._tzm ? this.zone(this._tzm) : "string" == typeof this._i && this.zone(this._i),
					this
				},
				hasAlignedHourOffset: function(e) {
					return e = e ? me(e).zone() : 0,
					(this.zone() - e) % 60 == 0
				},
				daysInMonth: function() {
					return S(this.year(), this.month())
				},
				dayOfYear: function(e) {
					var t = ve((me(this).startOf("day") - me(this).startOf("year")) / 864e5) + 1;
					return null == e ? t: this.add(e - t, "d")
				},
				quarter: function(e) {
					return null == e ? Math.ceil((this.month() + 1) / 3) : this.month(3 * (e - 1) + this.month() % 3)
				},
				weekYear: function(e) {
					var t = oe(this, this.localeData()._week.dow, this.localeData()._week.doy).year;
					return null == e ? t: this.add(e - t, "y")
				},
				isoWeekYear: function(e) {
					var t = oe(this, 1, 4).year;
					return null == e ? t: this.add(e - t, "y")
				},
				week: function(e) {
					var t = this.localeData().week(this);
					return null == e ? t: this.add(7 * (e - t), "d")
				},
				isoWeek: function(e) {
					var t = oe(this, 1, 4).week;
					return null == e ? t: this.add(7 * (e - t), "d")
				},
				weekday: function(e) {
					var t = (this.day() + 7 - this.localeData()._week.dow) % 7;
					return null == e ? t: this.add(e - t, "d")
				},
				isoWeekday: function(e) {
					return null == e ? this.day() || 7 : this.day(this.day() % 7 ? e: e - 7)
				},
				isoWeeksInYear: function() {
					return M(this.year(), 1, 4)
				},
				weeksInYear: function() {
					var e = this.localeData()._week;
					return M(this.year(), e.dow, e.doy)
				},
				get: function(e) {
					return e = x(e),
					this[e]()
				},
				set: function(e, t) {
					return e = x(e),
					"function" == typeof this[e] && this[e](t),
					this
				},
				locale: function(t) {
					var n;
					return t === e ? this._locale._abbr: (null != (n = me.localeData(t)) && (this._locale = n), this)
				},
				lang: o("moment().lang() is deprecated. Use moment().localeData() instead.",
				function(t) {
					return t === e ? this.localeData() : this.locale(t)
				}),
				localeData: function() {
					return this._locale
				},
				_dateTzOffset: function() {
					return 15 * Math.round(this._d.getTimezoneOffset() / 15)
				}
			}),
			me.fn.millisecond = me.fn.milliseconds = he("Milliseconds", !1),
			me.fn.second = me.fn.seconds = he("Seconds", !1),
			me.fn.minute = me.fn.minutes = he("Minutes", !1),
			me.fn.hour = me.fn.hours = he("Hours", !0),
			me.fn.date = he("Date", !0),
			me.fn.dates = o("dates accessor is deprecated. Use date instead.", he("Date", !0)),
			me.fn.year = he("FullYear", !0),
			me.fn.years = o("years accessor is deprecated. Use year instead.", he("FullYear", !0)),
			me.fn.days = me.fn.day,
			me.fn.months = me.fn.month,
			me.fn.weeks = me.fn.week,
			me.fn.isoWeeks = me.fn.isoWeek,
			me.fn.quarters = me.fn.quarter,
			me.fn.toJSON = me.fn.toISOString,
			d(me.duration.fn = u.prototype, {
				_bubble: function() {
					var e, t, n, i = this._milliseconds,
					r = this._days,
					o = this._months,
					a = this._data,
					s = 0;
					a.milliseconds = i % 1e3,
					e = f(i / 1e3),
					a.seconds = e % 60,
					t = f(e / 60),
					a.minutes = t % 60,
					n = f(t / 60),
					a.hours = n % 24,
					r += f(n / 24),
					o += f((r -= f(pe(s = f(fe(r))))) / 30),
					r %= 30,
					s += f(o / 12),
					o %= 12,
					a.days = r,
					a.months = o,
					a.years = s
				},
				abs: function() {
					return this._milliseconds = Math.abs(this._milliseconds),
					this._days = Math.abs(this._days),
					this._months = Math.abs(this._months),
					this._data.milliseconds = Math.abs(this._data.milliseconds),
					this._data.seconds = Math.abs(this._data.seconds),
					this._data.minutes = Math.abs(this._data.minutes),
					this._data.hours = Math.abs(this._data.hours),
					this._data.months = Math.abs(this._data.months),
					this._data.years = Math.abs(this._data.years),
					this
				},
				weeks: function() {
					return f(this.days() / 7)
				},
				valueOf: function() {
					return this._milliseconds + 864e5 * this._days + this._months % 12 * 2592e6 + 31536e6 * C(this._months / 12)
				},
				humanize: function(e) {
					var t = re(this, !e, this.localeData());
					return e && (t = this.localeData().pastFuture( + this, t)),
					this.localeData().postformat(t)
				},
				add: function(e, t) {
					var n = me.duration(e, t);
					return this._milliseconds += n._milliseconds,
					this._days += n._days,
					this._months += n._months,
					this._bubble(),
					this
				},
				subtract: function(e, t) {
					var n = me.duration(e, t);
					return this._milliseconds -= n._milliseconds,
					this._days -= n._days,
					this._months -= n._months,
					this._bubble(),
					this
				},
				get: function(e) {
					return e = x(e),
					this[e.toLowerCase() + "s"]()
				},
				as: function(e) {
					var t, n;
					if ("month" === (e = x(e)) || "year" === e) return t = this._days + this._milliseconds / 864e5,
					n = this._months + 12 * fe(t),
					"month" === e ? n: n / 12;
					switch (t = this._days + pe(this._months / 12), e) {
					case "week":
						return t / 7 + this._milliseconds / 6048e5;
					case "day":
						return t + this._milliseconds / 864e5;
					case "hour":
						return 24 * t + this._milliseconds / 36e5;
					case "minute":
						return 24 * t * 60 + this._milliseconds / 6e4;
					case "second":
						return 24 * t * 60 * 60 + this._milliseconds / 1e3;
					case "millisecond":
						return Math.floor(24 * t * 60 * 60 * 1e3) + this._milliseconds;
					default:
						throw new Error("Unknown unit " + e)
					}
				},
				lang: me.fn.lang,
				locale: me.fn.locale,
				toIsoString: o("toIsoString() is deprecated. Please use toISOString() instead (notice the capitals)",
				function() {
					return this.toISOString()
				}),
				toISOString: function() {
					var e = Math.abs(this.years()),
					t = Math.abs(this.months()),
					n = Math.abs(this.days()),
					i = Math.abs(this.hours()),
					r = Math.abs(this.minutes()),
					o = Math.abs(this.seconds() + this.milliseconds() / 1e3);
					return this.asSeconds() ? (this.asSeconds() < 0 ? "-": "") + "P" + (e ? e + "Y": "") + (t ? t + "M": "") + (n ? n + "D": "") + (i || r || o ? "T": "") + (i ? i + "H": "") + (r ? r + "M": "") + (o ? o + "S": "") : "P0D"
				},
				localeData: function() {
					return this._locale
				}
			}),
			me.duration.fn.toString = me.duration.fn.toISOString;
			for (ge in et) n(et, ge) &&
			function(e) {
				me.duration.fn[e] = function() {
					return this._data[e]
				}
			} (ge.toLowerCase());
			return me.duration.fn.asMilliseconds = function() {
				return this.as("ms")
			},
			me.duration.fn.asSeconds = function() {
				return this.as("s")
			},
			me.duration.fn.asMinutes = function() {
				return this.as("m")
			},
			me.duration.fn.asHours = function() {
				return this.as("h")
			},
			me.duration.fn.asDays = function() {
				return this.as("d")
			},
			me.duration.fn.asWeeks = function() {
				return this.as("weeks")
			},
			me.duration.fn.asMonths = function() {
				return this.as("M")
			},
			me.duration.fn.asYears = function() {
				return this.as("y")
			},
			me.locale("en", {
				ordinal: function(e) {
					var t = e % 10;
					return e + (1 === C(e % 100 / 10) ? "th": 1 === t ? "st": 2 === t ? "nd": 3 === t ? "rd": "th")
				}
			}),
			me
		}.call(this),
		e.Utils.moment = n,
		e.datepicker
	}),
	function(e) {
		var t;
		window.UIkit && (t = e(UIkit)),
		"function" == typeof define && define.amd && define("uikit-autocomplete", ["uikit"],
		function() {
			return t || e(UIkit)
		})
	} (function(e) {
		"use strict";
		var t;
		return e.component("autocomplete", {
			defaults: {
				minLength: 3,
				param: "search",
				method: "post",
				delay: 300,
				loadingClass: "uk-loading",
				flipDropdown: !1,
				skipClass: "uk-skip",
				hoverClass: "uk-active",
				source: null,
				renderer: null,
				template: '<ul class="uk-nav uk-nav-autocomplete uk-autocomplete-results">{{~items}}<li data-value="{{$item.value}}"><a>{{$item.value}}</a></li>{{/items}}</ul>'
			},
			visible: !1,
			value: null,
			selected: null,
			boot: function() {
				e.$html.on("focus.autocomplete.uikit", "[data-uk-autocomplete]",
				function(t) {
					var n = e.$(this);
					n.data("autocomplete") || e.autocomplete(n, e.Utils.options(n.attr("data-uk-autocomplete")))
				}),
				e.$html.on("click.autocomplete.uikit",
				function(e) {
					t && e.target != t.input[0] && t.hide()
				})
			},
			init: function() {
				var t = this,
				n = !1,
				i = e.Utils.debounce(function(e) {
					if (n) return n = !1;
					t.handle()
				},
				this.options.delay);
				this.dropdown = this.find(".uk-dropdown"),
				this.template = this.find('script[type="text/autocomplete"]').html(),
				this.template = e.Utils.template(this.template || this.options.template),
				this.input = this.find("input:first").attr("autocomplete", "off"),
				this.dropdown.length || (this.dropdown = e.$('<div class="uk-dropdown"></div>').appendTo(this.element)),
				this.options.flipDropdown && this.dropdown.addClass("uk-dropdown-flip"),
				this.dropdown.attr("aria-expanded", "false"),
				this.input.on({
					keydown: function(e) {
						if (e && e.which && !e.shiftKey && t.visible) switch (e.which) {
						case 13:
							n = !0,
							t.selected && (e.preventDefault(), t.select());
							break;
						case 38:
							e.preventDefault(),
							t.pick("prev", !0);
							break;
						case 40:
							e.preventDefault(),
							t.pick("next", !0);
							break;
						case 27:
						case 9:
							t.hide()
						}
					},
					keyup: i
				}),
				this.dropdown.on("click", ".uk-autocomplete-results > *",
				function() {
					t.select()
				}),
				this.dropdown.on("mouseover", ".uk-autocomplete-results > *",
				function() {
					t.pick(e.$(this))
				}),
				this.triggercomplete = i
			},
			handle: function() {
				var e = this,
				t = this.value;
				return this.value = this.input.val(),
				this.value.length < this.options.minLength ? this.hide() : (this.value != t && e.request(), this)
			},
			pick: function(t, n) {
				var i = this,
				r = e.$(this.dropdown.find(".uk-autocomplete-results").children(":not(." + this.options.skipClass + ")")),
				o = !1;
				if ("string" == typeof t || t.hasClass(this.options.skipClass)) {
					if ("next" == t || "prev" == t) {
						if (this.selected) {
							var a = r.index(this.selected);
							o = "next" == t ? r.eq(a + 1 < r.length ? a + 1 : 0) : r.eq(a - 1 < 0 ? r.length - 1 : a - 1)
						} else o = r["next" == t ? "first": "last"]();
						o = e.$(o)
					}
				} else o = t;
				if (o && o.length && (this.selected = o, r.removeClass(this.options.hoverClass), this.selected.addClass(this.options.hoverClass), n)) {
					var s = o.position().top,
					l = i.dropdown.scrollTop(); (s > i.dropdown.height() || s < 0) && i.dropdown.scrollTop(l + s)
				}
			},
			select: function() {
				if (this.selected) {
					var e = this.selected.data();
					this.trigger("selectitem.uk.autocomplete", [e, this]),
					e.value && this.input.val(e.value).trigger("change"),
					this.hide()
				}
			},
			show: function() {
				if (!this.visible) return this.visible = !0,
				this.element.addClass("uk-open"),
				t && t !== this && t.hide(),
				t = this,
				this.dropdown.attr("aria-expanded", "true"),
				this
			},
			hide: function() {
				if (this.visible) return this.visible = !1,
				this.element.removeClass("uk-open"),
				t === this && (t = !1),
				this.dropdown.attr("aria-expanded", "false"),
				this
			},
			request: function() {
				var t = this,
				n = function(e) {
					e && t.render(e),
					t.element.removeClass(t.options.loadingClass)
				};
				if (this.element.addClass(this.options.loadingClass), this.options.source) {
					var i = this.options.source;
					switch (typeof this.options.source) {
					case "function":
						this.options.source.apply(this, [n]);
						break;
					case "object":
						if (i.length) {
							var r = [];
							i.forEach(function(e) {
								e.value && -1 != e.value.toLowerCase().indexOf(t.value.toLowerCase()) && r.push(e)
							}),
							n(r)
						}
						break;
					case "string":
						var o = {};
						o[this.options.param] = this.value,
						e.$.ajax({
							url: this.options.source,
							data: o,
							type: this.options.method,
							dataType: "json"
						}).done(function(e) {
							n(e || [])
						});
						break;
					default:
						n(null)
					}
				} else this.element.removeClass(t.options.loadingClass)
			},
			render: function(e) {
				return this.dropdown.empty(),
				this.selected = !1,
				this.options.renderer ? this.options.renderer.apply(this, [e]) : e && e.length && (this.dropdown.append(this.template({
					items: e
				})), this.show(), this.trigger("show.uk.autocomplete")),
				this
			}
		}),
		e.autocomplete
	}),
	function(e) {
		var t;
		window.UIkit && (t = e(UIkit)),
		"function" == typeof define && define.amd && define("uikit-lightbox", ["uikit"],
		function() {
			return t || e(UIkit)
		})
	} (function(e) {
		"use strict";
		function t(t) {
			if (n) return n.lightbox = t,
			n; (n = e.$(['<div class="uk-modal">', '<div class="uk-modal-dialog uk-modal-dialog-lightbox uk-slidenav-position" style="margin-left:auto;margin-right:auto;width:200px;height:200px;top:' + Math.abs(window.innerHeight / 2 - 200) + 'px;">', '<a href="#" class="uk-modal-close uk-close uk-close-alt"></a>', '<div class="uk-lightbox-content"></div>', '<div class="uk-modal-spinner uk-hidden"></div>', "</div>", "</div>"].join("")).appendTo("body")).dialog = n.find(".uk-modal-dialog:first"),
			n.content = n.find(".uk-lightbox-content:first"),
			n.loader = n.find(".uk-modal-spinner:first"),
			n.closer = n.find(".uk-close.uk-close-alt"),
			n.modal = e.modal(n, {
				modal: !1
			}),
			n.on("swipeRight swipeLeft",
			function(e) {
				n.lightbox["swipeLeft" == e.type ? "next": "previous"]()
			}).on("click", "[data-lightbox-previous], [data-lightbox-next]",
			function(t) {
				t.preventDefault(),
				n.lightbox[e.$(this).is("[data-lightbox-next]") ? "next": "previous"]()
			}),
			n.on("hide.uk.modal",
			function(e) {
				n.content.html("")
			});
			var i = {
				w: window.innerWidth,
				h: window.innerHeight
			};
			return e.$win.on("load resize orientationchange", e.Utils.debounce(function(t) {
				i.w !== window.innerWidth && n.is(":visible") && !e.Utils.isFullscreen() && n.lightbox.fitSize(),
				i = {
					w: window.innerWidth,
					h: window.innerHeight
				}
			},
			100)),
			n.lightbox = t,
			n
		}
		var n, i = {};
		return e.component("lightbox", {
			defaults: {
				allowfullscreen: !0,
				duration: 400,
				group: !1,
				keyboard: !0
			},
			index: 0,
			items: !1,
			boot: function() {
				e.$html.on("click", "[data-uk-lightbox]",
				function(t) {
					t.preventDefault();
					var n = e.$(this);
					n.data("lightbox") || e.lightbox(n, e.Utils.options(n.attr("data-uk-lightbox"))),
					n.data("lightbox").show(n)
				}),
				e.$doc.on("keyup",
				function(e) {
					if (n && n.is(":visible") && n.lightbox.options.keyboard) switch (e.preventDefault(), e.keyCode) {
					case 37:
						n.lightbox.previous();
						break;
					case 39:
						n.lightbox.next()
					}
				})
			},
			init: function() {
				var t = [];
				if (this.index = 0, this.siblings = [], this.element && this.element.length) {
					var n = this.options.group ? e.$('[data-uk-lightbox*="' + this.options.group + '"]') : this.element;
					n.each(function() {
						var n = e.$(this);
						t.push({
							source: n.attr("href"),
							title: n.attr("data-title") || n.attr("title"),
							type: n.attr("data-lightbox-type") || "auto",
							link: n
						})
					}),
					this.index = n.index(this.element),
					this.siblings = t
				} else this.options.group && this.options.group.length && (this.siblings = this.options.group);
				this.trigger("lightbox-init", [this])
			},
			show: function(n) {
				this.modal = t(this),
				this.modal.dialog.stop(),
				this.modal.content.stop();
				var i, r, o = this,
				a = e.$.Deferred();
				"object" == typeof(n = n || 0) && this.siblings.forEach(function(e, t) {
					n[0] === e.link[0] && (n = t)
				}),
				n < 0 ? n = this.siblings.length - n: this.siblings[n] || (n = 0),
				r = this.siblings[n],
				i = {
					lightbox: o,
					source: r.source,
					type: r.type,
					index: n,
					promise: a,
					title: r.title,
					item: r,
					meta: {
						content: "",
						width: null,
						height: null
					}
				},
				this.index = n,
				this.modal.content.empty(),
				this.modal.is(":visible") || (this.modal.content.css({
					width: "",
					height: ""
				}).empty(), this.modal.modal.show()),
				this.modal.loader.removeClass("uk-hidden"),
				a.promise().done(function() {
					o.data = i,
					o.fitSize(i)
				}).fail(function() {
					i.meta.content = '<div class="uk-position-cover uk-flex uk-flex-middle uk-flex-center"><strong>Loading resource failed!</strong></div>',
					i.meta.width = 400,
					i.meta.height = 300,
					o.data = i,
					o.fitSize(i)
				}),
				o.trigger("showitem.uk.lightbox", [i])
			},
			fitSize: function() {
				var t = this,
				n = this.data,
				i = this.modal.dialog.outerWidth() - this.modal.dialog.width(),
				r = parseInt(this.modal.dialog.css("margin-top"), 10) + parseInt(this.modal.dialog.css("margin-bottom"), 10),
				o = n.meta.content,
				a = t.options.duration;
				this.siblings.length > 1 && (o = [o, '<a href="#" class="uk-slidenav uk-slidenav-contrast uk-slidenav-previous uk-hidden-touch" data-lightbox-previous></a>', '<a href="#" class="uk-slidenav uk-slidenav-contrast uk-slidenav-next uk-hidden-touch" data-lightbox-next></a>'].join(""));
				var s, l, c = e.$("<div>&nbsp;</div>").css({
					opacity: 0,
					position: "absolute",
					top: 0,
					left: 0,
					width: "100%",
					maxWidth: t.modal.dialog.css("max-width"),
					padding: t.modal.dialog.css("padding"),
					margin: t.modal.dialog.css("margin")
				}),
				u = n.meta.width,
				d = n.meta.height;
				c.appendTo("body").width(),
				s = c.width(),
				l = window.innerHeight - r,
				c.remove(),
				this.modal.dialog.find(".uk-modal-caption").remove(),
				n.title && (this.modal.dialog.append('<div class="uk-modal-caption">' + n.title + "</div>"), l -= this.modal.dialog.find(".uk-modal-caption").outerHeight()),
				s < n.meta.width && (d = Math.floor(d * (s / u)), u = s),
				l < d && (d = Math.floor(l), u = Math.ceil(n.meta.width * (l / n.meta.height))),
				this.modal.content.css("opacity", 0).width(u).html(o),
				"iframe" == n.type && this.modal.content.find("iframe:first").height(d);
				var h = d + i,
				f = Math.floor(window.innerHeight / 2 - h / 2) - r;
				f < 0 && (f = 0),
				this.modal.closer.addClass("uk-hidden"),
				t.modal.data("mwidth") == u && t.modal.data("mheight") == d && (a = 0),
				this.modal.dialog.animate({
					width: u + i,
					height: d + i,
					top: f
				},
				a, "swing",
				function() {
					t.modal.loader.addClass("uk-hidden"),
					t.modal.content.css({
						width: ""
					}).animate({
						opacity: 1
					},
					function() {
						t.modal.closer.removeClass("uk-hidden")
					}),
					t.modal.data({
						mwidth: u,
						mheight: d
					})
				})
			},
			next: function() {
				this.show(this.siblings[this.index + 1] ? this.index + 1 : 0)
			},
			previous: function() {
				this.show(this.siblings[this.index - 1] ? this.index - 1 : this.siblings.length - 1)
			}
		}),
		e.plugin("lightbox", "image", {
			init: function(e) {
				e.on("showitem.uk.lightbox",
				function(e, t) {
					if ("image" == t.type || t.source && t.source.match(/\.(jpg|jpeg|png|gif|svg)$/i)) {
						var n = function(e, n, i) {
							t.meta = {
								content: '<img class="uk-responsive-width" width="' + n + '" height="' + i + '" src ="' + e + '">',
								width: n,
								height: i
							},
							t.type = "image",
							t.promise.resolve()
						};
						if (i[t.source]) n(t.source, i[t.source].width, i[t.source].height);
						else {
							var r = new Image;
							r.onerror = function() {
								t.promise.reject("Loading image failed")
							},
							r.onload = function() {
								i[t.source] = {
									width: r.width,
									height: r.height
								},
								n(t.source, i[t.source].width, i[t.source].height)
							},
							r.src = t.source
						}
					}
				})
			}
		}),
		e.plugin("lightbox", "youtube", {
			init: function(e) {
				var t = /(\/\/.*?youtube\.[a-z]+)\/watch\?v=([^&]+)&?(.*)/,
				r = /youtu\.be\/(.*)/;
				e.on("showitem.uk.lightbox",
				function(e, o) {
					var a, s, l = function(e, t, i) {
						o.meta = {
							content: '<iframe src="//www.youtube.com/embed/' + e + '" width="' + t + '" height="' + i + '" style="max-width:100%;"' + (n.lightbox.options.allowfullscreen ? " allowfullscreen": "") + "></iframe>",
							width: t,
							height: i
						},
						o.type = "iframe",
						o.promise.resolve()
					};
					if ((s = o.source.match(t)) && (a = s[2]), (s = o.source.match(r)) && (a = s[1]), a) {
						if (i[a]) l(a, i[a].width, i[a].height);
						else {
							var c = new Image,
							u = !1;
							c.onerror = function() {
								i[a] = {
									width: 640,
									height: 320
								},
								l(a, i[a].width, i[a].height)
							},
							c.onload = function() {
								120 == c.width && 90 == c.height ? u ? (i[a] = {
									width: 640,
									height: 320
								},
								l(a, i[a].width, i[a].height)) : (u = !0, c.src = "//img.youtube.com/vi/" + a + "/0.jpg") : (i[a] = {
									width: c.width,
									height: c.height
								},
								l(a, c.width, c.height))
							},
							c.src = "//img.youtube.com/vi/" + a + "/maxresdefault.jpg"
						}
						e.stopImmediatePropagation()
					}
				})
			}
		}),
		e.plugin("lightbox", "vimeo", {
			init: function(t) {
				var r, o = /(\/\/.*?)vimeo\.[a-z]+\/([0-9]+).*?/;
				t.on("showitem.uk.lightbox",
				function(t, a) {
					var s, l = function(e, t, i) {
						a.meta = {
							content: '<iframe src="//player.vimeo.com/video/' + e + '" width="' + t + '" height="' + i + '" style="width:100%;box-sizing:border-box;"' + (n.lightbox.options.allowfullscreen ? " allowfullscreen": "") + "></iframe>",
							width: t,
							height: i
						},
						a.type = "iframe",
						a.promise.resolve()
					}; (r = a.source.match(o)) && (s = r[2], i[s] ? l(s, i[s].width, i[s].height) : e.$.ajax({
						type: "GET",
						url: "//vimeo.com/api/oembed.json?url=" + encodeURI(a.source),
						jsonp: "callback",
						dataType: "jsonp",
						success: function(e) {
							i[s] = {
								width: e.width,
								height: e.height
							},
							l(s, i[s].width, i[s].height)
						}
					}), t.stopImmediatePropagation())
				})
			}
		}),
		e.plugin("lightbox", "video", {
			init: function(t) {
				t.on("showitem.uk.lightbox",
				function(t, n) {
					var r = function(e, t, i) {
						n.meta = {
							content: '<video class="uk-responsive-width" src="' + e + '" width="' + t + '" height="' + i + '" controls></video>',
							width: t,
							height: i
						},
						n.type = "video",
						n.promise.resolve()
					};
					if ("video" == n.type || n.source.match(/\.(mp4|webm|ogv)$/i)) if (i[n.source]) r(n.source, i[n.source].width, i[n.source].height);
					else var o = e.$('<video style="position:fixed;visibility:hidden;top:-10000px;"></video>').attr("src", n.source).appendTo("body"),
					a = setInterval(function() {
						o[0].videoWidth && (clearInterval(a), i[n.source] = {
							width: o[0].videoWidth,
							height: o[0].videoHeight
						},
						r(n.source, i[n.source].width, i[n.source].height), o.remove())
					},
					20)
				})
			}
		}),
		UIkit.plugin("lightbox", "iframe", {
			init: function(e) {
				e.on("showitem.uk.lightbox",
				function(t, i) { ("iframe" === i.type || i.source.match(/\.(html|php)$/)) &&
					function(e, t, r) {
						i.meta = {
							content: '<iframe class="uk-responsive-width" src="' + e + '" width="' + t + '" height="' + r + '"' + (n.lightbox.options.allowfullscreen ? " allowfullscreen": "") + "></iframe>",
							width: t,
							height: r
						},
						i.type = "iframe",
						i.promise.resolve()
					} (i.source, e.options.width || 800, e.options.height || 600)
				})
			}
		}),
		e.lightbox.create = function(t, n) {
			if (t) {
				var i = [];
				return t.forEach(function(t) {
					i.push(e.$.extend({
						source: "",
						title: "",
						type: "auto",
						link: !1
					},
					"string" == typeof t ? {
						source: t
					}: t))
				}),
				e.lightbox(e.$.extend({},
				n, {
					group: i
				}))
			}
		},
		e.lightbox
	}),
	function(e) {
		var t;
		window.UIkit && (t = e(UIkit)),
		"function" == typeof define && define.amd && define("uikit-htmleditor", ["uikit"],
		function() {
			return t || e(UIkit)
		})
	} (function(e) {
		"use strict";
		var t = [];
		return e.component("htmleditor", {
			defaults: {
				iframe: !1,
				mode: "split",
				markdown: !1,
				autocomplete: !0,
				enablescripts: !1,
				height: 500,
				maxsplitsize: 1e3,
				codemirror: {
					mode: "htmlmixed",
					lineWrapping: !0,
					dragDrop: !1,
					autoCloseTags: !0,
					matchTags: !0,
					autoCloseBrackets: !0,
					matchBrackets: !0,
					indentUnit: 4,
					indentWithTabs: !1,
					tabSize: 4,
					hintOptions: {
						completionSingle: !1
					}
				},
				toolbar: ["bold", "italic", "strike", "link", "image", "blockquote", "listUl", "listOl"],
				lblPreview: "Preview",
				lblCodeview: "HTML",
				lblMarkedview: "Markdown"
			},
			boot: function() {
				e.ready(function(t) {
					e.$("textarea[data-uk-htmleditor]", t).each(function() {
						var t = e.$(this);
						t.data("htmleditor") || e.htmleditor(t, e.Utils.options(t.attr("data-uk-htmleditor")))
					})
				})
			},
			init: function() {
				var n = this,
				i = e.components.htmleditor.template;
				this.CodeMirror = this.options.CodeMirror || CodeMirror,
				this.buttons = {},
				i = (i = i.replace(/\{:lblPreview}/g, this.options.lblPreview)).replace(/\{:lblCodeview}/g, this.options.lblCodeview),
				this.htmleditor = e.$(i),
				this.content = this.htmleditor.find(".uk-htmleditor-content"),
				this.toolbar = this.htmleditor.find(".uk-htmleditor-toolbar"),
				this.preview = this.htmleditor.find(".uk-htmleditor-preview").children().eq(0),
				this.code = this.htmleditor.find(".uk-htmleditor-code"),
				this.element.before(this.htmleditor).appendTo(this.code),
				this.editor = this.CodeMirror.fromTextArea(this.element[0], this.options.codemirror),
				this.editor.htmleditor = this,
				this.editor.on("change", e.Utils.debounce(function() {
					n.render()
				},
				150)),
				this.editor.on("change",
				function() {
					n.editor.save(),
					n.element.trigger("input")
				}),
				this.code.find(".CodeMirror").css("height", this.options.height),
				this.options.iframe ? (this.iframe = e.$('<iframe class="uk-htmleditor-iframe" frameborder="0" scrolling="auto" height="100" width="100%"></iframe>'), this.preview.append(this.iframe), this.iframe[0].contentWindow.document.open(), this.iframe[0].contentWindow.document.close(), this.preview.container = e.$(this.iframe[0].contentWindow.document).find("body"), "string" == typeof this.options.iframe && this.preview.container.parent().append('<link rel="stylesheet" href="' + this.options.iframe + '">')) : this.preview.container = this.preview,
				e.$win.on("resize load", e.Utils.debounce(function() {
					n.fit()
				},
				200));
				var r = this.iframe ? this.preview.container: n.preview.parent(),
				o = this.code.find(".CodeMirror-sizer"),
				a = this.code.find(".CodeMirror-scroll").on("scroll", e.Utils.debounce(function() {
					if ("tab" != n.htmleditor.attr("data-mode")) {
						var e = o.height() - a.height(),
						t = (r[0].scrollHeight - (n.iframe ? n.iframe.height() : r.height())) / e,
						i = a.scrollTop() * t;
						r.scrollTop(i)
					}
				},
				10));
				this.htmleditor.on("click", ".uk-htmleditor-button-code, .uk-htmleditor-button-preview",
				function(t) {
					t.preventDefault(),
					"tab" == n.htmleditor.attr("data-mode") && (n.htmleditor.find(".uk-htmleditor-button-code, .uk-htmleditor-button-preview").removeClass("uk-active").filter(this).addClass("uk-active"), n.activetab = e.$(this).hasClass("uk-htmleditor-button-code") ? "code": "preview", n.htmleditor.attr("data-active-tab", n.activetab), n.editor.refresh())
				}),
				this.htmleditor.on("click", "a[data-htmleditor-button]",
				function() {
					n.code.is(":visible") && n.trigger("action." + e.$(this).data("htmleditor-button"), [n.editor])
				}),
				this.preview.parent().css("height", this.code.height()),
				this.options.autocomplete && this.CodeMirror.showHint && this.CodeMirror.hint && this.CodeMirror.hint.html && this.editor.on("inputRead", e.Utils.debounce(function() {
					var e = n.editor.getDoc().getCursor();
					if ("xml" == n.CodeMirror.innerMode(n.editor.getMode(), n.editor.getTokenAt(e).state).mode.name) {
						var t = n.editor.getCursor(),
						i = n.editor.getTokenAt(t);
						"<" != i.string.charAt(0) && "attribute" != i.type || n.CodeMirror.showHint(n.editor, n.CodeMirror.hint.html, {
							completeSingle: !1
						})
					}
				},
				100)),
				this.debouncedRedraw = e.Utils.debounce(function() {
					n.redraw()
				},
				5),
				this.on("init.uk.component",
				function() {
					n.debouncedRedraw()
				}),
				this.element.attr("data-uk-check-display", 1).on("display.uk.check",
				function(e) {
					this.htmleditor.is(":visible") && this.fit()
				}.bind(this)),
				t.push(this)
			},
			addButton: function(e, t) {
				this.buttons[e] = t
			},
			addButtons: function(t) {
				e.$.extend(this.buttons, t)
			},
			replaceInPreview: function(e, t) {
				function n(e) {
					var t = i.getValue().substring(0, e).split("\n");
					return {
						line: t.length - 1,
						ch: t[t.length - 1].length
					}
				}
				var i = this.editor,
				r = [],
				o = i.getValue(),
				a = -1,
				s = 0;
				return this.currentvalue = this.currentvalue.replace(e,
				function() {
					var e = {
						matches: arguments,
						from: n(a = o.indexOf(arguments[0], ++a)),
						to: n(a + arguments[0].length),
						replace: function(t) {
							i.replaceRange(t, e.from, e.to)
						},
						inRange: function(t) {
							return t.line === e.from.line && t.line === e.to.line ? t.ch >= e.from.ch && t.ch < e.to.ch: t.line === e.from.line && t.ch >= e.from.ch || t.line > e.from.line && t.line < e.to.line || t.line === e.to.line && t.ch < e.to.ch
						}
					},
					l = "string" == typeof t ? t: t(e, s);
					return l || "" === l ? (s++, r.push(e), l) : arguments[0]
				}),
				r
			},
			_buildtoolbar: function() {
				if (this.options.toolbar && this.options.toolbar.length) {
					var e = this,
					t = [];
					this.toolbar.empty(),
					this.options.toolbar.forEach(function(n) {
						if (e.buttons[n]) {
							var i = e.buttons[n].title ? e.buttons[n].title: n;
							t.push('<li><a data-htmleditor-button="' + n + '" title="' + i + '" data-uk-tooltip>' + e.buttons[n].label + "</a></li>")
						}
					}),
					this.toolbar.html(t.join("\n"))
				}
			},
			fit: function() {
				var e = this.options.mode;
				"split" == e && this.htmleditor.width() < this.options.maxsplitsize && (e = "tab"),
				"tab" == e && (this.activetab || (this.activetab = "code", this.htmleditor.attr("data-active-tab", this.activetab)), this.htmleditor.find(".uk-htmleditor-button-code, .uk-htmleditor-button-preview").removeClass("uk-active").filter("code" == this.activetab ? ".uk-htmleditor-button-code": ".uk-htmleditor-button-preview").addClass("uk-active")),
				this.editor.refresh(),
				this.preview.parent().css("height", this.code.height()),
				this.htmleditor.attr("data-mode", e)
			},
			redraw: function() {
				this._buildtoolbar(),
				this.render(),
				this.fit()
			},
			getMode: function() {
				return this.editor.getOption("mode")
			},
			getCursorMode: function() {
				var e = {
					mode: "html"
				};
				return this.trigger("cursorMode", [e]),
				e.mode
			},
			render: function() {
				if (this.currentvalue = this.editor.getValue(), this.options.enablescripts || (this.currentvalue = this.currentvalue.replace(/<(script|style)\b[^<]*(?:(?!<\/(script|style)>)<[^<]*)*<\/(script|style)>/gim, "")), !this.currentvalue) return this.element.val(""),
				void this.preview.container.html("");
				this.trigger("render", [this]),
				this.trigger("renderLate", [this]),
				this.preview.container.html(this.currentvalue)
			},
			addShortcut: function(t, n) {
				var i = {};
				return e.$.isArray(t) || (t = [t]),
				t.forEach(function(e) {
					i[e] = n
				}),
				this.editor.addKeyMap(i),
				i
			},
			addShortcutAction: function(e, t) {
				var n = this;
				this.addShortcut(t,
				function() {
					n.element.trigger("action." + e, [n.editor])
				})
			},
			replaceSelection: function(e) {
				var t = this.editor.getSelection();
				if (!t.length) {
					for (var n = this.editor.getCursor(), i = this.editor.getLine(n.line), r = n.ch, o = r; o < i.length && /[\w$]+/.test(i.charAt(o));)++o;
					for (; r && /[\w$]+/.test(i.charAt(r - 1));)--r;
					var a = r != o && i.slice(r, o);
					a && (this.editor.setSelection({
						line: n.line,
						ch: r
					},
					{
						line: n.line,
						ch: o
					}), t = a)
				}
				var s = e.replace("$1", t);
				this.editor.replaceSelection(s, "end"),
				this.editor.focus()
			},
			replaceLine: function(e) {
				var t = this.editor.getDoc().getCursor(),
				n = this.editor.getLine(t.line),
				i = e.replace("$1", n);
				this.editor.replaceRange(i, {
					line: t.line,
					ch: 0
				},
				{
					line: t.line,
					ch: n.length
				}),
				this.editor.setCursor({
					line: t.line,
					ch: i.length
				}),
				this.editor.focus()
			},
			save: function() {
				this.editor.save()
			}
		}),
		e.components.htmleditor.template = ['<div class="uk-htmleditor uk-clearfix" data-mode="split">', '<div class="uk-htmleditor-navbar">', '<ul class="uk-htmleditor-navbar-nav uk-htmleditor-toolbar"></ul>', '<div class="uk-htmleditor-navbar-flip">', '<ul class="uk-htmleditor-navbar-nav">', '<li class="uk-htmleditor-button-code"><a>{:lblCodeview}</a></li>', '<li class="uk-htmleditor-button-preview"><a>{:lblPreview}</a></li>', '<li><a data-htmleditor-button="fullscreen"><i class="uk-icon-expand"></i></a></li>', "</ul>", "</div>", "</div>", '<div class="uk-htmleditor-content">', '<div class="uk-htmleditor-code"></div>', '<div class="uk-htmleditor-preview"><div></div></div>', "</div>", "</div>"].join(""),
		e.plugin("htmleditor", "base", {
			init: function(t) {
				function n(e, n, i) {
					t.on("action." + e,
					function() {
						"html" == t.getCursorMode() && t["replaceLine" == i ? "replaceLine": "replaceSelection"](n)
					})
				}
				t.addButtons({
					fullscreen: {
						title: "Fullscreen",
						label: '<i class="uk-icon-expand"></i>'
					},
					bold: {
						title: "Bold",
						label: '<i class="uk-icon-bold"></i>'
					},
					italic: {
						title: "Italic",
						label: '<i class="uk-icon-italic"></i>'
					},
					strike: {
						title: "Strikethrough",
						label: '<i class="uk-icon-strikethrough"></i>'
					},
					blockquote: {
						title: "Blockquote",
						label: '<i class="uk-icon-quote-right"></i>'
					},
					link: {
						title: "Link",
						label: '<i class="uk-icon-link"></i>'
					},
					image: {
						title: "Image",
						label: '<i class="uk-icon-picture-o"></i>'
					},
					listUl: {
						title: "Unordered List",
						label: '<i class="uk-icon-list-ul"></i>'
					},
					listOl: {
						title: "Ordered List",
						label: '<i class="uk-icon-list-ol"></i>'
					}
				}),
				n("bold", "<strong>$1</strong>"),
				n("italic", "<em>$1</em>"),
				n("strike", "<del>$1</del>"),
				n("blockquote", "<blockquote><p>$1</p></blockquote>", "replaceLine"),
				n("link", '<a href="http://">$1</a>'),
				n("image", '<img src="http://" alt="$1">');
				var i = function(e) {
					if ("html" == t.getCursorMode()) {
						e = e || "ul";
						for (var n = t.editor,
						i = n.getDoc(), r = i.getCursor(!0), o = i.getCursor(!1), a = CodeMirror.innerMode(n.getMode(), n.getTokenAt(n.getCursor()).state), s = a && a.state && a.state.context && -1 != ["ul", "ol"].indexOf(a.state.context.tagName), l = r.line; l < o.line + 1; l++) n.replaceRange("<li>" + n.getLine(l) + "</li>", {
							line: l,
							ch: 0
						},
						{
							line: l,
							ch: n.getLine(l).length
						});
						s ? n.setCursor({
							line: o.line,
							ch: n.getLine(o.line).length
						}) : (n.replaceRange("<" + e + ">\n" + n.getLine(r.line), {
							line: r.line,
							ch: 0
						},
						{
							line: r.line,
							ch: n.getLine(r.line).length
						}), n.replaceRange(n.getLine(o.line + 1) + "\n</" + e + ">", {
							line: o.line + 1,
							ch: 0
						},
						{
							line: o.line + 1,
							ch: n.getLine(o.line + 1).length
						}), n.setCursor({
							line: o.line + 1,
							ch: n.getLine(o.line + 1).length
						})),
						n.focus()
					}
				};
				t.on("action.listUl",
				function() {
					i("ul")
				}),
				t.on("action.listOl",
				function() {
					i("ol")
				}),
				t.htmleditor.on("click", 'a[data-htmleditor-button="fullscreen"]',
				function() {
					t.htmleditor.toggleClass("uk-htmleditor-fullscreen");
					var n = t.editor.getWrapperElement();
					if (t.htmleditor.hasClass("uk-htmleditor-fullscreen")) {
						var i = !1,
						r = t.htmleditor.parents().each(function() {
							"fixed" != e.$(this).css("position") || e.$(this).is("html") || (i = e.$(this))
						});
						if (t.htmleditor.data("fixedParents", !1), i) {
							var o = [];
							i = i.parent().find(r).each(function() {
								"none" != e.$(this).css("transform") && o.push(e.$(this).data("transform-reset", {
									transform: this.style.transform,
									"-webkit-transform": this.style.webkitTransform,
									"-webkit-transition": this.style.webkitTransition,
									transition: this.style.transition
								}).css({
									transform: "none",
									"-webkit-transform": "none",
									"-webkit-transition": "none",
									transition: "none"
								}))
							}),
							t.htmleditor.data("fixedParents", o)
						}
						t.editor.state.fullScreenRestore = {
							scrollTop: window.pageYOffset,
							scrollLeft: window.pageXOffset,
							width: n.style.width,
							height: n.style.height
						},
						n.style.width = "",
						n.style.height = t.content.height() + "px",
						document.documentElement.style.overflow = "hidden"
					} else {
						document.documentElement.style.overflow = "";
						var a = t.editor.state.fullScreenRestore;
						n.style.width = a.width,
						n.style.height = a.height,
						window.scrollTo(a.scrollLeft, a.scrollTop),
						t.htmleditor.data("fixedParents") && t.htmleditor.data("fixedParents").forEach(function(e) {
							e.css(e.data("transform-reset"))
						})
					}
					setTimeout(function() {
						t.fit(),
						e.$win.trigger("resize")
					},
					50)
				}),
				t.addShortcut(["Ctrl-S", "Cmd-S"],
				function() {
					t.element.trigger("htmleditor-save", [t])
				}),
				t.addShortcutAction("bold", ["Ctrl-B", "Cmd-B"])
			}
		}),
		e.plugin("htmleditor", "markdown", {
			init: function(t) {
				function n() {
					t.editor.setOption("mode", "gfm"),
					t.htmleditor.find(".uk-htmleditor-button-code a").html(t.options.lblMarkedview)
				}
				function i(e, n, i) {
					t.on("action." + e,
					function() {
						"markdown" == t.getCursorMode() && t["replaceLine" == i ? "replaceLine": "replaceSelection"](n)
					})
				}
				var r = t.options.mdparser || window.marked || null;
				r && (t.options.markdown && n(), i("bold", "**$1**"), i("italic", "*$1*"), i("strike", "~~$1~~"), i("blockquote", "> $1", "replaceLine"), i("link", "[$1](http://)"), t.on("action.image",
				function() {
					uploadImage(function(e, n) {
						e || t.replaceSelection("\n\n![" + n.name + "](" + n.url + ")\n\n")
					})
				}), t.on("action.listUl",
				function() {
					if ("markdown" == t.getCursorMode()) {
						for (var e = t.editor,
						n = e.getDoc().getCursor(!0), i = e.getDoc().getCursor(!1), r = n.line; r < i.line + 1; r++) e.replaceRange("* " + e.getLine(r), {
							line: r,
							ch: 0
						},
						{
							line: r,
							ch: e.getLine(r).length
						});
						e.setCursor({
							line: i.line,
							ch: e.getLine(i.line).length
						}),
						e.focus()
					}
				}), t.on("action.listOl",
				function() {
					if ("markdown" == t.getCursorMode()) {
						var e = t.editor,
						n = e.getDoc().getCursor(!0),
						i = e.getDoc().getCursor(!1),
						r = 1;
						if (n.line > 0) {
							var o; (o = e.getLine(n.line - 1).match(/^(\d+)\./)) && (r = Number(o[1]) + 1)
						}
						for (var a = n.line; a < i.line + 1; a++) e.replaceRange(r + ". " + e.getLine(a), {
							line: a,
							ch: 0
						},
						{
							line: a,
							ch: e.getLine(a).length
						}),
						r++;
						e.setCursor({
							line: i.line,
							ch: e.getLine(i.line).length
						}),
						e.focus()
					}
				}), t.on("renderLate",
				function() {
					"gfm" == t.editor.options.mode && (t.currentvalue = r(t.currentvalue))
				}), t.on("cursorMode",
				function(e, n) {
					if ("gfm" == t.editor.options.mode) {
						var i = t.editor.getDoc().getCursor();
						t.editor.getTokenAt(i).state.base.htmlState || (n.mode = "markdown")
					}
				}), e.$.extend(t, {
					enableMarkdown: function() {
						n(),
						this.render()
					},
					disableMarkdown: function() {
						this.editor.setOption("mode", "htmlmixed"),
						this.htmleditor.find(".uk-htmleditor-button-code a").html(this.options.lblCodeview),
						this.render()
					}
				}), t.on({
					enableMarkdown: function() {
						t.enableMarkdown()
					},
					disableMarkdown: function() {
						t.disableMarkdown()
					}
				}))
			}
		}),
		e.htmleditor
	}),
	function(e) {
		if ("object" == typeof exports && "object" == typeof module) module.exports = e();
		else {
			if ("function" == typeof define && define.amd) return define([], e); (this || window).CodeMirror = e()
		}
	} (function() {
		"use strict";
		function e(n, i) {
			if (! (this instanceof e)) return new e(n, i);
			this.options = i = i ? $r(i) : {},
			$r(Zo, i, !1),
			h(i);
			var r = i.value;
			"string" == typeof r && (r = new wa(r, i.mode, null, i.lineSeparator)),
			this.doc = r;
			var o = new e.inputStyles[i.inputStyle](this),
			s = this.display = new t(n, r, o);
			s.wrapper.CodeMirror = this,
			l(this),
			a(this),
			i.lineWrapping && (this.display.wrapper.className += " CodeMirror-wrap"),
			i.autofocus && !Mo && s.input.focus(),
			g(this),
			this.state = {
				keyMaps: [],
				overlays: [],
				modeGen: 0,
				overwrite: !1,
				delayingBlurEvent: !1,
				focused: !1,
				suppressEdits: !1,
				pasteIncoming: !1,
				cutIncoming: !1,
				selectingText: !1,
				draggingText: !1,
				highlight: new Tr,
				keySeq: null,
				specialChars: null
			};
			var c = this;
			go && 11 > vo && setTimeout(function() {
				c.display.input.reset(!0)
			},
			20),
			Ht(this),
			Br(),
			bt(this),
			this.curOp.forceUpdate = !0,
			Gi(this, r),
			i.autofocus && !Mo || c.hasFocus() ? setTimeout(qr(mn, this), 20) : gn(this);
			for (var u in Xo) Xo.hasOwnProperty(u) && Xo[u](this, i[u], Qo);
			k(this),
			i.finishInit && i.finishInit(this);
			for (var d = 0; d < na.length; ++d) na[d](this);
			kt(this),
			yo && i.lineWrapping && "optimizelegibility" == getComputedStyle(s.lineDiv).textRendering && (s.lineDiv.style.textRendering = "auto")
		}
		function t(e, t, n) {
			var i = this;
			this.input = n,
			i.scrollbarFiller = zr("div", null, "CodeMirror-scrollbar-filler"),
			i.scrollbarFiller.setAttribute("cm-not-content", "true"),
			i.gutterFiller = zr("div", null, "CodeMirror-gutter-filler"),
			i.gutterFiller.setAttribute("cm-not-content", "true"),
			i.lineDiv = zr("div", null, "CodeMirror-code"),
			i.selectionDiv = zr("div", null, null, "position: relative; z-index: 1"),
			i.cursorDiv = zr("div", null, "CodeMirror-cursors"),
			i.measure = zr("div", null, "CodeMirror-measure"),
			i.lineMeasure = zr("div", null, "CodeMirror-measure"),
			i.lineSpace = zr("div", [i.measure, i.lineMeasure, i.selectionDiv, i.cursorDiv, i.lineDiv], null, "position: relative; outline: none"),
			i.mover = zr("div", [zr("div", [i.lineSpace], "CodeMirror-lines")], null, "position: relative"),
			i.sizer = zr("div", [i.mover], "CodeMirror-sizer"),
			i.sizerWidth = null,
			i.heightForcer = zr("div", null, null, "position: absolute; height: " + Na + "px; width: 1px;"),
			i.gutters = zr("div", null, "CodeMirror-gutters"),
			i.lineGutter = null,
			i.scroller = zr("div", [i.sizer, i.heightForcer, i.gutters], "CodeMirror-scroll"),
			i.scroller.setAttribute("tabIndex", "-1"),
			i.wrapper = zr("div", [i.scrollbarFiller, i.gutterFiller, i.scroller], "CodeMirror"),
			go && 8 > vo && (i.gutters.style.zIndex = -1, i.scroller.style.paddingRight = 0),
			yo || fo && Mo || (i.scroller.draggable = !0),
			e && (e.appendChild ? e.appendChild(i.wrapper) : e(i.wrapper)),
			i.viewFrom = i.viewTo = t.first,
			i.reportedViewFrom = i.reportedViewTo = t.first,
			i.view = [],
			i.renderedView = null,
			i.externalMeasured = null,
			i.viewOffset = 0,
			i.lastWrapHeight = i.lastWrapWidth = 0,
			i.updateLineNumbers = null,
			i.nativeBarWidth = i.barHeight = i.barWidth = 0,
			i.scrollbarsClipped = !1,
			i.lineNumWidth = i.lineNumInnerWidth = i.lineNumChars = null,
			i.alignWidgets = !1,
			i.cachedCharWidth = i.cachedTextHeight = i.cachedPaddingH = null,
			i.maxLine = null,
			i.maxLineLength = 0,
			i.maxLineChanged = !1,
			i.wheelDX = i.wheelDY = i.wheelStartX = i.wheelStartY = null,
			i.shift = !1,
			i.selForContextMenu = null,
			i.activeTouch = null,
			n.init(i)
		}
		function n(t) {
			t.doc.mode = e.getMode(t.options, t.doc.modeOption),
			i(t)
		}
		function i(e) {
			e.doc.iter(function(e) {
				e.stateAfter && (e.stateAfter = null),
				e.styles && (e.styles = null)
			}),
			e.doc.frontier = e.doc.first,
			Ie(e, 100),
			e.state.modeGen++,
			e.curOp && $t(e)
		}
		function r(e) {
			var t = vt(e.display),
			n = e.options.lineWrapping,
			i = n && Math.max(5, e.display.scroller.clientWidth / yt(e.display) - 3);
			return function(r) {
				if (wi(e.doc, r)) return 0;
				var o = 0;
				if (r.widgets) for (var a = 0; a < r.widgets.length; a++) r.widgets[a].height && (o += r.widgets[a].height);
				return n ? o + (Math.ceil(r.text.length / i) || 1) * t: o + t
			}
		}
		function o(e) {
			var t = e.doc,
			n = r(e);
			t.iter(function(e) {
				var t = n(e);
				t != e.height && Xi(e, t)
			})
		}
		function a(e) {
			e.display.wrapper.className = e.display.wrapper.className.replace(/\s*cm-s-\S+/g, "") + e.options.theme.replace(/(^|\s)\s*/g, " cm-s-"),
			at(e)
		}
		function s(e) {
			l(e),
			$t(e),
			setTimeout(function() {
				w(e)
			},
			20)
		}
		function l(e) {
			var t = e.display.gutters,
			n = e.options.gutters;
			Wr(t);
			for (var i = 0; i < n.length; ++i) {
				var r = n[i],
				o = t.appendChild(zr("div", null, "CodeMirror-gutter " + r));
				"CodeMirror-linenumbers" == r && (e.display.lineGutter = o, o.style.width = (e.display.lineNumWidth || 1) + "px")
			}
			t.style.display = i ? "": "none",
			c(e)
		}
		function c(e) {
			var t = e.display.gutters.offsetWidth;
			e.display.sizer.style.marginLeft = t + "px"
		}
		function u(e) {
			if (0 == e.height) return 0;
			for (var t, n = e.text.length,
			i = e; t = fi(i);) i = (r = t.find(0, !0)).from.line,
			n += r.from.ch - r.to.ch;
			for (i = e; t = pi(i);) {
				var r = t.find(0, !0);
				n -= i.text.length - r.from.ch,
				n += (i = r.to.line).text.length - r.to.ch
			}
			return n
		}
		function d(e) {
			var t = e.display,
			n = e.doc;
			t.maxLine = Vi(n, n.first),
			t.maxLineLength = u(t.maxLine),
			t.maxLineChanged = !0,
			n.iter(function(e) {
				var n = u(e);
				n > t.maxLineLength && (t.maxLineLength = n, t.maxLine = e)
			})
		}
		function h(e) {
			var t = Or(e.gutters, "CodeMirror-linenumbers"); - 1 == t && e.lineNumbers ? e.gutters = e.gutters.concat(["CodeMirror-linenumbers"]) : t > -1 && !e.lineNumbers && (e.gutters = e.gutters.slice(0), e.gutters.splice(t, 1))
		}
		function f(e) {
			var t = e.display,
			n = t.gutters.offsetWidth,
			i = Math.round(e.doc.height + Ye(e.display));
			return {
				clientHeight: t.scroller.clientHeight,
				viewHeight: t.wrapper.clientHeight,
				scrollWidth: t.scroller.scrollWidth,
				clientWidth: t.scroller.clientWidth,
				viewWidth: t.wrapper.clientWidth,
				barLeft: e.options.fixedGutter ? n: 0,
				docHeight: i,
				scrollHeight: i + Ue(e) + t.barHeight,
				nativeBarWidth: t.nativeBarWidth,
				gutterWidth: n
			}
		}
		function p(e, t, n) {
			this.cm = n;
			var i = this.vert = zr("div", [zr("div", null, null, "min-width: 1px")], "CodeMirror-vscrollbar"),
			r = this.horiz = zr("div", [zr("div", null, null, "height: 100%; min-height: 1px")], "CodeMirror-hscrollbar");
			e(i),
			e(r),
			Ma(i, "scroll",
			function() {
				i.clientHeight && t(i.scrollTop, "vertical")
			}),
			Ma(r, "scroll",
			function() {
				r.clientWidth && t(r.scrollLeft, "horizontal")
			}),
			this.checkedZeroWidth = !1,
			go && 8 > vo && (this.horiz.style.minHeight = this.vert.style.minWidth = "18px")
		}
		function m() {}
		function g(t) {
			t.display.scrollbars && (t.display.scrollbars.clear(), t.display.scrollbars.addClass && Ga(t.display.wrapper, t.display.scrollbars.addClass)),
			t.display.scrollbars = new e.scrollbarModel[t.options.scrollbarStyle](function(e) {
				t.display.wrapper.insertBefore(e, t.display.scrollbarFiller),
				Ma(e, "mousedown",
				function() {
					t.state.focused && setTimeout(function() {
						t.display.input.focus()
					},
					0)
				}),
				e.setAttribute("cm-not-content", "true")
			},
			function(e, n) {
				"horizontal" == n ? nn(t, e) : tn(t, e)
			},
			t),
			t.display.scrollbars.addClass && Va(t.display.wrapper, t.display.scrollbars.addClass)
		}
		function v(e, t) {
			t || (t = f(e));
			var n = e.display.barWidth,
			i = e.display.barHeight;
			y(e, t);
			for (var r = 0; 4 > r && n != e.display.barWidth || i != e.display.barHeight; r++) n != e.display.barWidth && e.options.lineWrapping && O(e),
			y(e, f(e)),
			n = e.display.barWidth,
			i = e.display.barHeight
		}
		function y(e, t) {
			var n = e.display,
			i = n.scrollbars.update(t);
			n.sizer.style.paddingRight = (n.barWidth = i.right) + "px",
			n.sizer.style.paddingBottom = (n.barHeight = i.bottom) + "px",
			n.heightForcer.style.borderBottom = i.bottom + "px solid transparent",
			i.right && i.bottom ? (n.scrollbarFiller.style.display = "block", n.scrollbarFiller.style.height = i.bottom + "px", n.scrollbarFiller.style.width = i.right + "px") : n.scrollbarFiller.style.display = "",
			i.bottom && e.options.coverGutterNextToScrollbar && e.options.fixedGutter ? (n.gutterFiller.style.display = "block", n.gutterFiller.style.height = i.bottom + "px", n.gutterFiller.style.width = t.gutterWidth + "px") : n.gutterFiller.style.display = ""
		}
		function b(e, t, n) {
			var i = n && null != n.top ? Math.max(0, n.top) : e.scroller.scrollTop;
			i = Math.floor(i - Fe(e));
			var r = n && null != n.bottom ? n.bottom: i + e.wrapper.clientHeight,
			o = Ji(t, i),
			a = Ji(t, r);
			if (n && n.ensure) {
				var s = n.ensure.from.line,
				l = n.ensure.to.line;
				o > s ? (o = s, a = Ji(t, er(Vi(t, s)) + e.wrapper.clientHeight)) : Math.min(l, t.lastLine()) >= a && (o = Ji(t, er(Vi(t, l)) - e.wrapper.clientHeight), a = l)
			}
			return {
				from: o,
				to: Math.max(a, o + 1)
			}
		}
		function w(e) {
			var t = e.display,
			n = t.view;
			if (t.alignWidgets || t.gutters.firstChild && e.options.fixedGutter) {
				for (var i = _(t) - t.scroller.scrollLeft + e.doc.scrollLeft, r = t.gutters.offsetWidth, o = i + "px", a = 0; a < n.length; a++) if (!n[a].hidden) {
					e.options.fixedGutter && n[a].gutter && (n[a].gutter.style.left = o);
					var s = n[a].alignable;
					if (s) for (var l = 0; l < s.length; l++) s[l].style.left = o
				}
				e.options.fixedGutter && (t.gutters.style.left = i + r + "px")
			}
		}
		function k(e) {
			if (!e.options.lineNumbers) return ! 1;
			var t = e.doc,
			n = x(e.options, t.first + t.size - 1),
			i = e.display;
			if (n.length != i.lineNumChars) {
				var r = i.measure.appendChild(zr("div", [zr("div", n)], "CodeMirror-linenumber CodeMirror-gutter-elt")),
				o = r.firstChild.offsetWidth,
				a = r.offsetWidth - o;
				return i.lineGutter.style.width = "",
				i.lineNumInnerWidth = Math.max(o, i.lineGutter.offsetWidth - a) + 1,
				i.lineNumWidth = i.lineNumInnerWidth + a,
				i.lineNumChars = i.lineNumInnerWidth ? n.length: -1,
				i.lineGutter.style.width = i.lineNumWidth + "px",
				c(e),
				!0
			}
			return ! 1
		}
		function x(e, t) {
			return String(e.lineNumberFormatter(t + e.firstLineNumber))
		}
		function _(e) {
			return e.scroller.getBoundingClientRect().left - e.sizer.getBoundingClientRect().left
		}
		function C(e, t, n) {
			var i = e.display;
			this.viewport = t,
			this.visible = b(i, e.doc, t),
			this.editorIsHidden = !i.wrapper.offsetWidth,
			this.wrapperHeight = i.wrapper.clientHeight,
			this.wrapperWidth = i.wrapper.clientWidth,
			this.oldDisplayWidth = Be(e),
			this.force = n,
			this.dims = A(e),
			this.events = []
		}
		function S(e) {
			var t = e.display; ! t.scrollbarsClipped && t.scroller.offsetWidth && (t.nativeBarWidth = t.scroller.offsetWidth - t.scroller.clientWidth, t.heightForcer.style.height = Ue(e) + "px", t.sizer.style.marginBottom = -t.nativeBarWidth + "px", t.sizer.style.borderRightWidth = Ue(e) + "px", t.scrollbarsClipped = !0)
		}
		function M(e, t) {
			var n = e.display,
			i = e.doc;
			if (t.editorIsHidden) return jt(e),
			!1;
			if (!t.force && t.visible.from >= n.viewFrom && t.visible.to <= n.viewTo && (null == n.updateLineNumbers || n.updateLineNumbers >= n.viewTo) && n.renderedView == n.view && 0 == Wt(e)) return ! 1;
			k(e) && (jt(e), t.dims = A(e));
			var r = i.first + i.size,
			o = Math.max(t.visible.from - e.options.viewportMargin, i.first),
			a = Math.min(r, t.visible.to + e.options.viewportMargin);
			n.viewFrom < o && o - n.viewFrom < 20 && (o = Math.max(i.first, n.viewFrom)),
			n.viewTo > a && n.viewTo - a < 20 && (a = Math.min(r, n.viewTo)),
			Eo && (o = yi(e.doc, o), a = bi(e.doc, a));
			var s = o != n.viewFrom || a != n.viewTo || n.lastWrapHeight != t.wrapperHeight || n.lastWrapWidth != t.wrapperWidth;
			zt(e, o, a),
			n.viewOffset = er(Vi(e.doc, n.viewFrom)),
			e.display.mover.style.top = n.viewOffset + "px";
			var l = Wt(e);
			if (!s && 0 == l && !t.force && n.renderedView == n.view && (null == n.updateLineNumbers || n.updateLineNumbers >= n.viewTo)) return ! 1;
			var c = Fr();
			return l > 4 && (n.lineDiv.style.display = "none"),
			E(e, n.updateLineNumbers, t.dims),
			l > 4 && (n.lineDiv.style.display = ""),
			n.renderedView = n.view,
			c && Fr() != c && c.offsetHeight && c.focus(),
			Wr(n.cursorDiv),
			Wr(n.selectionDiv),
			n.gutters.style.height = n.sizer.style.minHeight = 0,
			s && (n.lastWrapHeight = t.wrapperHeight, n.lastWrapWidth = t.wrapperWidth, Ie(e, 400)),
			n.updateLineNumbers = null,
			!0
		}
		function T(e, t) {
			for (var n = t.viewport,
			i = !0; (i && e.options.lineWrapping && t.oldDisplayWidth != Be(e) || (n && null != n.top && (n = {
				top: Math.min(e.doc.height + Ye(e.display) - Ge(e), n.top)
			}), t.visible = b(e.display, e.doc, n), !(t.visible.from >= e.display.viewFrom && t.visible.to <= e.display.viewTo))) && M(e, t); i = !1) {
				O(e);
				var r = f(e);
				Ee(e),
				v(e, r),
				L(e, r)
			}
			t.signal(e, "update", e),
			(e.display.viewFrom != e.display.reportedViewFrom || e.display.viewTo != e.display.reportedViewTo) && (t.signal(e, "viewportChange", e, e.display.viewFrom, e.display.viewTo), e.display.reportedViewFrom = e.display.viewFrom, e.display.reportedViewTo = e.display.viewTo)
		}
		function D(e, t) {
			var n = new C(e, t);
			if (M(e, n)) {
				O(e),
				T(e, n);
				var i = f(e);
				Ee(e),
				v(e, i),
				L(e, i),
				n.finish()
			}
		}
		function L(e, t) {
			e.display.sizer.style.minHeight = t.docHeight + "px",
			e.display.heightForcer.style.top = t.docHeight + "px",
			e.display.gutters.style.height = t.docHeight + e.display.barHeight + Ue(e) + "px"
		}
		function O(e) {
			for (var t = e.display,
			n = t.lineDiv.offsetTop,
			i = 0; i < t.view.length; i++) {
				var r, o = t.view[i];
				if (!o.hidden) {
					if (go && 8 > vo) {
						var a = o.node.offsetTop + o.node.offsetHeight;
						r = a - n,
						n = a
					} else {
						var s = o.node.getBoundingClientRect();
						r = s.bottom - s.top
					}
					var l = o.line.height - r;
					if (2 > r && (r = vt(t)), (l > .001 || -.001 > l) && (Xi(o.line, r), N(o.line), o.rest)) for (var c = 0; c < o.rest.length; c++) N(o.rest[c])
				}
			}
		}
		function N(e) {
			if (e.widgets) for (var t = 0; t < e.widgets.length; ++t) e.widgets[t].height = e.widgets[t].node.parentNode.offsetHeight
		}
		function A(e) {
			for (var t = e.display,
			n = {},
			i = {},
			r = t.gutters.clientLeft,
			o = t.gutters.firstChild,
			a = 0; o; o = o.nextSibling, ++a) n[e.options.gutters[a]] = o.offsetLeft + o.clientLeft + r,
			i[e.options.gutters[a]] = o.clientWidth;
			return {
				fixedPos: _(t),
				gutterTotalWidth: t.gutters.offsetWidth,
				gutterLeft: n,
				gutterWidth: i,
				wrapperWidth: t.wrapper.clientWidth
			}
		}
		function E(e, t, n) {
			function i(t) {
				var n = t.nextSibling;
				return yo && To && e.display.currentWheelTarget == t ? t.style.display = "none": t.parentNode.removeChild(t),
				n
			}
			for (var r = e.display,
			o = e.options.lineNumbers,
			a = r.lineDiv,
			s = a.firstChild,
			l = r.view,
			c = r.viewFrom,
			u = 0; u < l.length; u++) {
				var d = l[u];
				if (d.hidden);
				else if (d.node && d.node.parentNode == a) {
					for (; s != d.node;) s = i(s);
					var h = o && null != t && c >= t && d.lineNumber;
					d.changes && (Or(d.changes, "gutter") > -1 && (h = !1), $(e, d, c, n)),
					h && (Wr(d.lineNumber), d.lineNumber.appendChild(document.createTextNode(x(e.options, c)))),
					s = d.node.nextSibling
				} else {
					var f = F(e, d, c, n);
					a.insertBefore(f, s)
				}
				c += d.size
			}
			for (; s;) s = i(s)
		}
		function $(e, t, n, i) {
			for (var r = 0; r < t.changes.length; r++) {
				var o = t.changes[r];
				"text" == o ? I(e, t) : "gutter" == o ? W(e, t, n, i) : "class" == o ? z(t) : "widget" == o && H(e, t, i)
			}
			t.changes = null
		}
		function q(e) {
			return e.node == e.text && (e.node = zr("div", null, null, "position: relative"), e.text.parentNode && e.text.parentNode.replaceChild(e.node, e.text), e.node.appendChild(e.text), go && 8 > vo && (e.node.style.zIndex = 2)),
			e.node
		}
		function j(e) {
			var t = e.bgClass ? e.bgClass + " " + (e.line.bgClass || "") : e.line.bgClass;
			if (t && (t += " CodeMirror-linebackground"), e.background) t ? e.background.className = t: (e.background.parentNode.removeChild(e.background), e.background = null);
			else if (t) {
				var n = q(e);
				e.background = n.insertBefore(zr("div", null, t), n.firstChild)
			}
		}
		function P(e, t) {
			var n = e.display.externalMeasured;
			return n && n.line == t.line ? (e.display.externalMeasured = null, t.measure = n.measure, n.built) : ji(e, t)
		}
		function I(e, t) {
			var n = t.text.className,
			i = P(e, t);
			t.text == t.node && (t.node = i.pre),
			t.text.parentNode.replaceChild(i.pre, t.text),
			t.text = i.pre,
			i.bgClass != t.bgClass || i.textClass != t.textClass ? (t.bgClass = i.bgClass, t.textClass = i.textClass, z(t)) : n && (t.text.className = n)
		}
		function z(e) {
			j(e),
			e.line.wrapClass ? q(e).className = e.line.wrapClass: e.node != e.text && (e.node.className = "");
			var t = e.textClass ? e.textClass + " " + (e.line.textClass || "") : e.line.textClass;
			e.text.className = t || ""
		}
		function W(e, t, n, i) {
			if (t.gutter && (t.node.removeChild(t.gutter), t.gutter = null), t.gutterBackground && (t.node.removeChild(t.gutterBackground), t.gutterBackground = null), t.line.gutterClass) {
				o = q(t);
				t.gutterBackground = zr("div", null, "CodeMirror-gutter-background " + t.line.gutterClass, "left: " + (e.options.fixedGutter ? i.fixedPos: -i.gutterTotalWidth) + "px; width: " + i.gutterTotalWidth + "px"),
				o.insertBefore(t.gutterBackground, t.text)
			}
			var r = t.line.gutterMarkers;
			if (e.options.lineNumbers || r) {
				var o = q(t),
				a = t.gutter = zr("div", null, "CodeMirror-gutter-wrapper", "left: " + (e.options.fixedGutter ? i.fixedPos: -i.gutterTotalWidth) + "px");
				if (e.display.input.setUneditable(a), o.insertBefore(a, t.text), t.line.gutterClass && (a.className += " " + t.line.gutterClass), !e.options.lineNumbers || r && r["CodeMirror-linenumbers"] || (t.lineNumber = a.appendChild(zr("div", x(e.options, n), "CodeMirror-linenumber CodeMirror-gutter-elt", "left: " + i.gutterLeft["CodeMirror-linenumbers"] + "px; width: " + e.display.lineNumInnerWidth + "px"))), r) for (var s = 0; s < e.options.gutters.length; ++s) {
					var l = e.options.gutters[s],
					c = r.hasOwnProperty(l) && r[l];
					c && a.appendChild(zr("div", [c], "CodeMirror-gutter-elt", "left: " + i.gutterLeft[l] + "px; width: " + i.gutterWidth[l] + "px"))
				}
			}
		}
		function H(e, t, n) {
			t.alignable && (t.alignable = null);
			for (var i = t.node.firstChild; i; i = r) {
				var r = i.nextSibling;
				"CodeMirror-linewidget" == i.className && t.node.removeChild(i)
			}
			Y(e, t, n)
		}
		function F(e, t, n, i) {
			var r = P(e, t);
			return t.text = t.node = r.pre,
			r.bgClass && (t.bgClass = r.bgClass),
			r.textClass && (t.textClass = r.textClass),
			z(t),
			W(e, t, n, i),
			Y(e, t, i),
			t.node
		}
		function Y(e, t, n) {
			if (R(e, t.line, t, n, !0), t.rest) for (var i = 0; i < t.rest.length; i++) R(e, t.rest[i], t, n, !1)
		}
		function R(e, t, n, i, r) {
			if (t.widgets) for (var o = q(n), a = 0, s = t.widgets; a < s.length; ++a) {
				var l = s[a],
				c = zr("div", [l.node], "CodeMirror-linewidget");
				l.handleMouseEvents || c.setAttribute("cm-ignore-events", "true"),
				U(l, c, n, i),
				e.display.input.setUneditable(c),
				r && l.above ? o.insertBefore(c, n.gutter || n.text) : o.appendChild(c),
				kr(l, "redraw")
			}
		}
		function U(e, t, n, i) {
			if (e.noHScroll) { (n.alignable || (n.alignable = [])).push(t);
				var r = i.wrapperWidth;
				t.style.left = i.fixedPos + "px",
				e.coverGutter || (r -= i.gutterTotalWidth, t.style.paddingLeft = i.gutterTotalWidth + "px"),
				t.style.width = r + "px"
			}
			e.coverGutter && (t.style.zIndex = 5, t.style.position = "relative", e.noHScroll || (t.style.marginLeft = -i.gutterTotalWidth + "px"))
		}
		function B(e) {
			return $o(e.line, e.ch)
		}
		function G(e, t) {
			return qo(e, t) < 0 ? t: e
		}
		function V(e, t) {
			return qo(e, t) < 0 ? e: t
		}
		function K(e) {
			e.state.focused || (e.display.input.focus(), mn(e))
		}
		function Z(e, t, n, i, r) {
			var o = e.doc;
			e.display.shift = !1,
			i || (i = o.sel);
			var a = e.state.pasteIncoming || "paste" == r,
			s = o.splitLines(t),
			l = null;
			if (a && i.ranges.length > 1) if (jo && jo.join("\n") == t) {
				if (i.ranges.length % jo.length == 0) {
					l = [];
					for (c = 0; c < jo.length; c++) l.push(o.splitLines(jo[c]))
				}
			} else s.length == i.ranges.length && (l = Nr(s,
			function(e) {
				return [e]
			}));
			for (var c = i.ranges.length - 1; c >= 0; c--) {
				var u = i.ranges[c],
				d = u.from(),
				h = u.to();
				u.empty() && (n && n > 0 ? d = $o(d.line, d.ch - n) : e.state.overwrite && !a && (h = $o(h.line, Math.min(Vi(o, h.line).text.length, h.ch + Lr(s).length))));
				var f = e.curOp.updateInput,
				p = {
					from: d,
					to: h,
					text: l ? l[c % l.length] : s,
					origin: r || (a ? "paste": e.state.cutIncoming ? "cut": "+input")
				};
				Cn(e.doc, p),
				kr(e, "inputRead", e, p)
			}
			t && !a && Q(e, t),
			jn(e),
			e.curOp.updateInput = f,
			e.curOp.typing = !0,
			e.state.pasteIncoming = e.state.cutIncoming = !1
		}
		function X(e, t) {
			var n = e.clipboardData && e.clipboardData.getData("text/plain");
			return n ? (e.preventDefault(), t.isReadOnly() || t.options.disableInput || Dt(t,
			function() {
				Z(t, n, 0, null, "paste")
			}), !0) : void 0
		}
		function Q(e, t) {
			if (e.options.electricChars && e.options.smartIndent) for (var n = e.doc.sel,
			i = n.ranges.length - 1; i >= 0; i--) {
				var r = n.ranges[i];
				if (! (r.head.ch > 100 || i && n.ranges[i - 1].head.line == r.head.line)) {
					var o = e.getModeAt(r.head),
					a = !1;
					if (o.electricChars) {
						for (var s = 0; s < o.electricChars.length; s++) if (t.indexOf(o.electricChars.charAt(s)) > -1) {
							a = In(e, r.head.line, "smart");
							break
						}
					} else o.electricInput && o.electricInput.test(Vi(e.doc, r.head.line).text.slice(0, r.head.ch)) && (a = In(e, r.head.line, "smart"));
					a && kr(e, "electricInput", e, r.head.line)
				}
			}
		}
		function J(e) {
			for (var t = [], n = [], i = 0; i < e.doc.sel.ranges.length; i++) {
				var r = e.doc.sel.ranges[i].head.line,
				o = {
					anchor: $o(r, 0),
					head: $o(r + 1, 0)
				};
				n.push(o),
				t.push(e.getRange(o.anchor, o.head))
			}
			return {
				text: t,
				ranges: n
			}
		}
		function ee(e) {
			e.setAttribute("autocorrect", "off"),
			e.setAttribute("autocapitalize", "off"),
			e.setAttribute("spellcheck", "false")
		}
		function te(e) {
			this.cm = e,
			this.prevInput = "",
			this.pollingFast = !1,
			this.polling = new Tr,
			this.inaccurateSelection = !1,
			this.hasSelection = !1,
			this.composing = null
		}
		function ne() {
			var e = zr("textarea", null, null, "position: absolute; padding: 0; width: 1px; height: 1em; outline: none"),
			t = zr("div", [e], null, "overflow: hidden; position: relative; width: 3px; height: 0px;");
			return yo ? e.style.width = "1000px": e.setAttribute("wrap", "off"),
			So && (e.style.border = "1px solid black"),
			ee(e),
			t
		}
		function ie(e) {
			this.cm = e,
			this.lastAnchorNode = this.lastAnchorOffset = this.lastFocusNode = this.lastFocusOffset = null,
			this.polling = new Tr,
			this.gracePeriod = !1
		}
		function re(e, t) {
			var n = Qe(e, t.line);
			if (!n || n.hidden) return null;
			var i = Vi(e.doc, t.line),
			r = Ke(n, i, t.line),
			o = tr(i),
			a = "left";
			o && (a = ao(o, t.ch) % 2 ? "right": "left");
			var s = tt(r.map, t.ch, a);
			return s.offset = "right" == s.collapse ? s.end: s.start,
			s
		}
		function oe(e, t) {
			return t && (e.bad = !0),
			e
		}
		function ae(e, t, n) {
			var i;
			if (t == e.display.lineDiv) {
				if (! (i = e.display.lineDiv.childNodes[n])) return oe(e.clipPos($o(e.display.viewTo - 1)), !0);
				t = null,
				n = 0
			} else for (i = t;; i = i.parentNode) {
				if (!i || i == e.display.lineDiv) return null;
				if (i.parentNode && i.parentNode == e.display.lineDiv) break
			}
			for (var r = 0; r < e.display.view.length; r++) {
				var o = e.display.view[r];
				if (o.node == i) return se(o, t, n)
			}
		}
		function se(e, t, n) {
			function i(t, n, i) {
				for (var r = -1; r < (u ? u.length: 0); r++) for (var o = 0 > r ? c.map: u[r], a = 0; a < o.length; a += 3) {
					var s = o[a + 2];
					if (s == t || s == n) {
						var l = Qi(0 > r ? e.line: e.rest[r]),
						d = o[a] + i;
						return (0 > i || s != t) && (d = o[a + (i ? 1 : 0)]),
						$o(l, d)
					}
				}
			}
			var r = e.text.firstChild,
			o = !1;
			if (!t || !Ra(r, t)) return oe($o(Qi(e.line), 0), !0);
			if (t == r && (o = !0, t = r.childNodes[n], n = 0, !t)) {
				var a = e.rest ? Lr(e.rest) : e.line;
				return oe($o(Qi(a), a.text.length), o)
			}
			var s = 3 == t.nodeType ? t: null,
			l = t;
			for (s || 1 != t.childNodes.length || 3 != t.firstChild.nodeType || (s = t.firstChild, n && (n = s.nodeValue.length)); l.parentNode != r;) l = l.parentNode;
			var c = e.measure,
			u = c.maps,
			d = i(s, l, n);
			if (d) return oe(d, o);
			for (var h = l.nextSibling,
			f = s ? s.nodeValue.length - n: 0; h; h = h.nextSibling) {
				if (d = i(h, h.firstChild, 0)) return oe($o(d.line, d.ch - f), o);
				f += h.textContent.length
			}
			for (var p = l.previousSibling,
			f = n; p; p = p.previousSibling) {
				if (d = i(p, p.firstChild, -1)) return oe($o(d.line, d.ch + f), o);
				f += h.textContent.length
			}
		}
		function le(e, t, n, i, r) {
			function o(e) {
				return function(t) {
					return t.id == e
				}
			}
			function a(t) {
				if (1 == t.nodeType) {
					var n = t.getAttribute("cm-text");
					if (null != n) return "" == n && (n = t.textContent.replace(/\u200b/g, "")),
					void(s += n);
					var u, d = t.getAttribute("cm-marker");
					if (d) {
						var h = e.findMarks($o(i, 0), $o(r + 1, 0), o( + d));
						return void(h.length && (u = h[0].find()) && (s += Ki(e.doc, u.from, u.to).join(c)))
					}
					if ("false" == t.getAttribute("contenteditable")) return;
					for (var f = 0; f < t.childNodes.length; f++) a(t.childNodes[f]);
					/^(pre|div|p)$/i.test(t.nodeName) && (l = !0)
				} else if (3 == t.nodeType) {
					var p = t.nodeValue;
					if (!p) return;
					l && (s += c, l = !1),
					s += p
				}
			}
			for (var s = "",
			l = !1,
			c = e.doc.lineSeparator(); a(t), t != n;) t = t.nextSibling;
			return s
		}
		function ce(e, t) {
			this.ranges = e,
			this.primIndex = t
		}
		function ue(e, t) {
			this.anchor = e,
			this.head = t
		}
		function de(e, t) {
			var n = e[t];
			e.sort(function(e, t) {
				return qo(e.from(), t.from())
			}),
			t = Or(e, n);
			for (var i = 1; i < e.length; i++) {
				var r = e[i],
				o = e[i - 1];
				if (qo(o.to(), r.from()) >= 0) {
					var a = V(o.from(), r.from()),
					s = G(o.to(), r.to()),
					l = o.empty() ? r.from() == r.head: o.from() == o.head;
					t >= i && --t,
					e.splice(--i, 2, new ue(l ? s: a, l ? a: s))
				}
			}
			return new ce(e, t)
		}
		function he(e, t) {
			return new ce([new ue(e, t || e)], 0)
		}
		function fe(e, t) {
			return Math.max(e.first, Math.min(t, e.first + e.size - 1))
		}
		function pe(e, t) {
			if (t.line < e.first) return $o(e.first, 0);
			var n = e.first + e.size - 1;
			return t.line > n ? $o(n, Vi(e, n).text.length) : me(t, Vi(e, t.line).text.length)
		}
		function me(e, t) {
			var n = e.ch;
			return null == n || n > t ? $o(e.line, t) : 0 > n ? $o(e.line, 0) : e
		}
		function ge(e, t) {
			return t >= e.first && t < e.first + e.size
		}
		function ve(e, t) {
			for (var n = [], i = 0; i < t.length; i++) n[i] = pe(e, t[i]);
			return n
		}
		function ye(e, t, n, i) {
			if (e.cm && e.cm.display.shift || e.extend) {
				var r = t.anchor;
				if (i) {
					var o = qo(n, r) < 0;
					o != qo(i, r) < 0 ? (r = n, n = i) : o != qo(n, i) < 0 && (n = i)
				}
				return new ue(r, n)
			}
			return new ue(i || n, n)
		}
		function be(e, t, n, i) {
			Se(e, new ce([ye(e, e.sel.primary(), t, n)], 0), i)
		}
		function we(e, t, n) {
			for (var i = [], r = 0; r < e.sel.ranges.length; r++) i[r] = ye(e, e.sel.ranges[r], t[r], null);
			Se(e, de(i, e.sel.primIndex), n)
		}
		function ke(e, t, n, i) {
			var r = e.sel.ranges.slice(0);
			r[t] = n,
			Se(e, de(r, e.sel.primIndex), i)
		}
		function xe(e, t, n, i) {
			Se(e, he(t, n), i)
		}
		function _e(e, t, n) {
			var i = {
				ranges: t.ranges,
				update: function(t) {
					this.ranges = [];
					for (var n = 0; n < t.length; n++) this.ranges[n] = new ue(pe(e, t[n].anchor), pe(e, t[n].head))
				},
				origin: n && n.origin
			};
			return La(e, "beforeSelectionChange", e, i),
			e.cm && La(e.cm, "beforeSelectionChange", e.cm, i),
			i.ranges != t.ranges ? de(i.ranges, i.ranges.length - 1) : t
		}
		function Ce(e, t, n) {
			var i = e.history.done,
			r = Lr(i);
			r && r.ranges ? (i[i.length - 1] = t, Me(e, t, n)) : Se(e, t, n)
		}
		function Se(e, t, n) {
			Me(e, t, n),
			lr(e, e.sel, e.cm ? e.cm.curOp.id: NaN, n)
		}
		function Me(e, t, n) { (Sr(e, "beforeSelectionChange") || e.cm && Sr(e.cm, "beforeSelectionChange")) && (t = _e(e, t, n)),
			Te(e, Le(e, t, n && n.bias || (qo(t.primary().head, e.sel.primary().head) < 0 ? -1 : 1), !0)),
			n && !1 === n.scroll || !e.cm || jn(e.cm)
		}
		function Te(e, t) {
			t.equals(e.sel) || (e.sel = t, e.cm && (e.cm.curOp.updateInput = e.cm.curOp.selectionChanged = !0, Cr(e.cm)), kr(e, "cursorActivity", e))
		}
		function De(e) {
			Te(e, Le(e, e.sel, null, !1), Ea)
		}
		function Le(e, t, n, i) {
			for (var r, o = 0; o < t.ranges.length; o++) {
				var a = t.ranges[o],
				s = t.ranges.length == e.sel.ranges.length && e.sel.ranges[o],
				l = Ne(e, a.anchor, s && s.anchor, n, i),
				c = Ne(e, a.head, s && s.head, n, i); (r || l != a.anchor || c != a.head) && (r || (r = t.ranges.slice(0, o)), r[o] = new ue(l, c))
			}
			return r ? de(r, t.primIndex) : t
		}
		function Oe(e, t, n, i, r) {
			var o = Vi(e, t.line);
			if (o.markedSpans) for (var a = 0; a < o.markedSpans.length; ++a) {
				var s = o.markedSpans[a],
				l = s.marker;
				if ((null == s.from || (l.inclusiveLeft ? s.from <= t.ch: s.from < t.ch)) && (null == s.to || (l.inclusiveRight ? s.to >= t.ch: s.to > t.ch))) {
					if (r && (La(l, "beforeCursorEnter"), l.explicitlyCleared)) {
						if (o.markedSpans) {--a;
							continue
						}
						break
					}
					if (!l.atomic) continue;
					if (n) {
						var c, u = l.find(0 > i ? 1 : -1);
						if ((0 > i ? l.inclusiveRight: l.inclusiveLeft) && (u = Ae(e, u, -i, u && u.line == t.line ? o: null)), u && u.line == t.line && (c = qo(u, n)) && (0 > i ? 0 > c: c > 0)) return Oe(e, u, t, i, r)
					}
					var d = l.find(0 > i ? -1 : 1);
					return (0 > i ? l.inclusiveLeft: l.inclusiveRight) && (d = Ae(e, d, i, d.line == t.line ? o: null)),
					d ? Oe(e, d, t, i, r) : null
				}
			}
			return t
		}
		function Ne(e, t, n, i, r) {
			var o = i || 1,
			a = Oe(e, t, n, o, r) || !r && Oe(e, t, n, o, !0) || Oe(e, t, n, -o, r) || !r && Oe(e, t, n, -o, !0);
			return a || (e.cantEdit = !0, $o(e.first, 0))
		}
		function Ae(e, t, n, i) {
			return 0 > n && 0 == t.ch ? t.line > e.first ? pe(e, $o(t.line - 1)) : null: n > 0 && t.ch == (i || Vi(e, t.line)).text.length ? t.line < e.first + e.size - 1 ? $o(t.line + 1, 0) : null: new $o(t.line, t.ch + n)
		}
		function Ee(e) {
			e.display.input.showSelection(e.display.input.prepareSelection())
		}
		function $e(e, t) {
			for (var n = e.doc,
			i = {},
			r = i.cursors = document.createDocumentFragment(), o = i.selection = document.createDocumentFragment(), a = 0; a < n.sel.ranges.length; a++) if (!1 !== t || a != n.sel.primIndex) {
				var s = n.sel.ranges[a];
				if (! (s.from().line >= e.display.viewTo || s.to().line < e.display.viewFrom)) {
					var l = s.empty(); (l || e.options.showCursorWhenSelecting) && qe(e, s.head, r),
					l || je(e, s, o)
				}
			}
			return i
		}
		function qe(e, t, n) {
			var i = ht(e, t, "div", null, null, !e.options.singleCursorHeightPerLine),
			r = n.appendChild(zr("div", " ", "CodeMirror-cursor"));
			if (r.style.left = i.left + "px", r.style.top = i.top + "px", r.style.height = Math.max(0, i.bottom - i.top) * e.options.cursorHeight + "px", i.other) {
				var o = n.appendChild(zr("div", " ", "CodeMirror-cursor CodeMirror-secondarycursor"));
				o.style.display = "",
				o.style.left = i.other.left + "px",
				o.style.top = i.other.top + "px",
				o.style.height = .85 * (i.other.bottom - i.other.top) + "px"
			}
		}
		function je(e, t, n) {
			function i(e, t, n, i) {
				0 > t && (t = 0),
				t = Math.round(t),
				i = Math.round(i),
				s.appendChild(zr("div", null, "CodeMirror-selected", "position: absolute; left: " + e + "px; top: " + t + "px; width: " + (null == n ? u - e: n) + "px; height: " + (i - t) + "px"))
			}
			function r(t, n, r) {
				function o(n, i) {
					return dt(e, $o(t, n), "div", d, i)
				}
				var s, l, d = Vi(a, t),
				h = d.text.length;
				return Xr(tr(d), n || 0, null == r ? h: r,
				function(e, t, a) {
					var d, f, p, m = o(e, "left");
					if (e == t) d = m,
					f = p = m.left;
					else {
						if (d = o(t - 1, "right"), "rtl" == a) {
							var g = m;
							m = d,
							d = g
						}
						f = m.left,
						p = d.right
					}
					null == n && 0 == e && (f = c),
					d.top - m.top > 3 && (i(f, m.top, null, m.bottom), f = c, m.bottom < d.top && i(f, m.bottom, null, d.top)),
					null == r && t == h && (p = u),
					(!s || m.top < s.top || m.top == s.top && m.left < s.left) && (s = m),
					(!l || d.bottom > l.bottom || d.bottom == l.bottom && d.right > l.right) && (l = d),
					c + 1 > f && (f = c),
					i(f, d.top, p - f, d.bottom)
				}),
				{
					start: s,
					end: l
				}
			}
			var o = e.display,
			a = e.doc,
			s = document.createDocumentFragment(),
			l = Re(e.display),
			c = l.left,
			u = Math.max(o.sizerWidth, Be(e) - o.sizer.offsetLeft) - l.right,
			d = t.from(),
			h = t.to();
			if (d.line == h.line) r(d.line, d.ch, h.ch);
			else {
				var f = Vi(a, d.line),
				p = Vi(a, h.line),
				m = gi(f) == gi(p),
				g = r(d.line, d.ch, m ? f.text.length + 1 : null).end,
				v = r(h.line, m ? 0 : null, h.ch).start;
				m && (g.top < v.top - 2 ? (i(g.right, g.top, null, g.bottom), i(c, v.top, v.left, v.bottom)) : i(g.right, g.top, v.left - g.right, g.bottom)),
				g.bottom < v.top && i(c, g.bottom, null, v.top)
			}
			n.appendChild(s)
		}
		function Pe(e) {
			if (e.state.focused) {
				var t = e.display;
				clearInterval(t.blinker);
				var n = !0;
				t.cursorDiv.style.visibility = "",
				e.options.cursorBlinkRate > 0 ? t.blinker = setInterval(function() {
					t.cursorDiv.style.visibility = (n = !n) ? "": "hidden"
				},
				e.options.cursorBlinkRate) : e.options.cursorBlinkRate < 0 && (t.cursorDiv.style.visibility = "hidden")
			}
		}
		function Ie(e, t) {
			e.doc.mode.startState && e.doc.frontier < e.display.viewTo && e.state.highlight.set(t, qr(ze, e))
		}
		function ze(e) {
			var t = e.doc;
			if (t.frontier < t.first && (t.frontier = t.first), !(t.frontier >= e.display.viewTo)) {
				var n = +new Date + e.options.workTime,
				i = ra(t.mode, He(e, t.frontier)),
				r = [];
				t.iter(t.frontier, Math.min(t.first + t.size, e.display.viewTo + 500),
				function(o) {
					if (t.frontier >= e.display.viewFrom) {
						var a = o.styles,
						s = o.text.length > e.options.maxHighlightLength,
						l = Ai(e, o, s ? ra(t.mode, i) : i, !0);
						o.styles = l.styles;
						var c = o.styleClasses,
						u = l.classes;
						u ? o.styleClasses = u: c && (o.styleClasses = null);
						for (var d = !a || a.length != o.styles.length || c != u && (!c || !u || c.bgClass != u.bgClass || c.textClass != u.textClass), h = 0; ! d && h < a.length; ++h) d = a[h] != o.styles[h];
						d && r.push(t.frontier),
						o.stateAfter = s ? i: ra(t.mode, i)
					} else o.text.length <= e.options.maxHighlightLength && $i(e, o.text, i),
					o.stateAfter = t.frontier % 5 == 0 ? ra(t.mode, i) : null;
					return++t.frontier,
					+new Date > n ? (Ie(e, e.options.workDelay), !0) : void 0
				}),
				r.length && Dt(e,
				function() {
					for (var t = 0; t < r.length; t++) qt(e, r[t], "text")
				})
			}
		}
		function We(e, t, n) {
			for (var i, r, o = e.doc,
			a = n ? -1 : t - (e.doc.mode.innerMode ? 1e3: 100), s = t; s > a; --s) {
				if (s <= o.first) return o.first;
				var l = Vi(o, s - 1);
				if (l.stateAfter && (!n || s <= o.frontier)) return s;
				var c = ja(l.text, null, e.options.tabSize); (null == r || i > c) && (r = s - 1, i = c)
			}
			return r
		}
		function He(e, t, n) {
			var i = e.doc,
			r = e.display;
			if (!i.mode.startState) return ! 0;
			var o = We(e, t, n),
			a = o > i.first && Vi(i, o - 1).stateAfter;
			return a = a ? ra(i.mode, a) : oa(i.mode),
			i.iter(o, t,
			function(n) {
				$i(e, n.text, a);
				var s = o == t - 1 || o % 5 == 0 || o >= r.viewFrom && o < r.viewTo;
				n.stateAfter = s ? ra(i.mode, a) : null,
				++o
			}),
			n && (i.frontier = o),
			a
		}
		function Fe(e) {
			return e.lineSpace.offsetTop
		}
		function Ye(e) {
			return e.mover.offsetHeight - e.lineSpace.offsetHeight
		}
		function Re(e) {
			if (e.cachedPaddingH) return e.cachedPaddingH;
			var t = Hr(e.measure, zr("pre", "x")),
			n = window.getComputedStyle ? window.getComputedStyle(t) : t.currentStyle,
			i = {
				left: parseInt(n.paddingLeft),
				right: parseInt(n.paddingRight)
			};
			return isNaN(i.left) || isNaN(i.right) || (e.cachedPaddingH = i),
			i
		}
		function Ue(e) {
			return Na - e.display.nativeBarWidth
		}
		function Be(e) {
			return e.display.scroller.clientWidth - Ue(e) - e.display.barWidth
		}
		function Ge(e) {
			return e.display.scroller.clientHeight - Ue(e) - e.display.barHeight
		}
		function Ve(e, t, n) {
			var i = e.options.lineWrapping,
			r = i && Be(e);
			if (!t.measure.heights || i && t.measure.width != r) {
				var o = t.measure.heights = [];
				if (i) {
					t.measure.width = r;
					for (var a = t.text.firstChild.getClientRects(), s = 0; s < a.length - 1; s++) {
						var l = a[s],
						c = a[s + 1];
						Math.abs(l.bottom - c.bottom) > 2 && o.push((l.bottom + c.top) / 2 - n.top)
					}
				}
				o.push(n.bottom - n.top)
			}
		}
		function Ke(e, t, n) {
			if (e.line == t) return {
				map: e.measure.map,
				cache: e.measure.cache
			};
			for (i = 0; i < e.rest.length; i++) if (e.rest[i] == t) return {
				map: e.measure.maps[i],
				cache: e.measure.caches[i]
			};
			for (var i = 0; i < e.rest.length; i++) if (Qi(e.rest[i]) > n) return {
				map: e.measure.maps[i],
				cache: e.measure.caches[i],
				before: !0
			}
		}
		function Ze(e, t) {
			var n = Qi(t = gi(t)),
			i = e.display.externalMeasured = new At(e.doc, t, n);
			i.lineN = n;
			var r = i.built = ji(e, i);
			return i.text = r.pre,
			Hr(e.display.lineMeasure, r.pre),
			i
		}
		function Xe(e, t, n, i) {
			return et(e, Je(e, t), n, i)
		}
		function Qe(e, t) {
			if (t >= e.display.viewFrom && t < e.display.viewTo) return e.display.view[Pt(e, t)];
			var n = e.display.externalMeasured;
			return n && t >= n.lineN && t < n.lineN + n.size ? n: void 0
		}
		function Je(e, t) {
			var n = Qi(t),
			i = Qe(e, n);
			i && !i.text ? i = null: i && i.changes && ($(e, i, n, A(e)), e.curOp.forceUpdate = !0),
			i || (i = Ze(e, t));
			var r = Ke(i, t, n);
			return {
				line: t,
				view: i,
				rect: null,
				map: r.map,
				cache: r.cache,
				before: r.before,
				hasHeights: !1
			}
		}
		function et(e, t, n, i, r) {
			t.before && (n = -1);
			var o, a = n + (i || "");
			return t.cache.hasOwnProperty(a) ? o = t.cache[a] : (t.rect || (t.rect = t.view.text.getBoundingClientRect()), t.hasHeights || (Ve(e, t.view, t.rect), t.hasHeights = !0), (o = nt(e, t, n, i)).bogus || (t.cache[a] = o)),
			{
				left: o.left,
				right: o.right,
				top: r ? o.rtop: o.top,
				bottom: r ? o.rbottom: o.bottom
			}
		}
		function tt(e, t, n) {
			for (var i, r, o, a, s = 0; s < e.length; s += 3) {
				var l = e[s],
				c = e[s + 1];
				if (l > t ? (r = 0, o = 1, a = "left") : c > t ? (r = t - l, o = r + 1) : (s == e.length - 3 || t == c && e[s + 3] > t) && (o = c - l, r = o - 1, t >= c && (a = "right")), null != r) {
					if (i = e[s + 2], l == c && n == (i.insertLeft ? "left": "right") && (a = n), "left" == n && 0 == r) for (; s && e[s - 2] == e[s - 3] && e[s - 1].insertLeft;) i = e[2 + (s -= 3)],
					a = "left";
					if ("right" == n && r == c - l) for (; s < e.length - 3 && e[s + 3] == e[s + 4] && !e[s + 5].insertLeft;) i = e[(s += 3) + 2],
					a = "right";
					break
				}
			}
			return {
				node: i,
				start: r,
				end: o,
				collapse: a,
				coverStart: l,
				coverEnd: c
			}
		}
		function nt(e, t, n, i) {
			var r, o = tt(t.map, n, i),
			a = o.node,
			s = o.start,
			l = o.end,
			c = o.collapse;
			if (3 == a.nodeType) {
				for (g = 0; 4 > g; g++) {
					for (; s && Ir(t.line.text.charAt(o.coverStart + s));)--s;
					for (; o.coverStart + l < o.coverEnd && Ir(t.line.text.charAt(o.coverStart + l));)++l;
					if ((r = go && 9 > vo && 0 == s && l == o.coverEnd - o.coverStart ? a.parentNode.getBoundingClientRect() : go && e.options.lineWrapping ? (u = Wa(a, s, l).getClientRects()).length ? u["right" == i ? u.length - 1 : 0] : Wo: Wa(a, s, l).getBoundingClientRect() || Wo).left || r.right || 0 == s) break;
					l = s,
					s -= 1,
					c = "right"
				}
				go && 11 > vo && (r = it(e.display.measure, r))
			} else {
				s > 0 && (c = i = "right");
				var u;
				r = e.options.lineWrapping && (u = a.getClientRects()).length > 1 ? u["right" == i ? u.length - 1 : 0] : a.getBoundingClientRect()
			}
			if (go && 9 > vo && !s && (!r || !r.left && !r.right)) {
				var d = a.parentNode.getClientRects()[0];
				r = d ? {
					left: d.left,
					right: d.left + yt(e.display),
					top: d.top,
					bottom: d.bottom
				}: Wo
			}
			for (var h = r.top - t.rect.top,
			f = r.bottom - t.rect.top,
			p = (h + f) / 2, m = t.view.measure.heights, g = 0; g < m.length - 1 && !(p < m[g]); g++);
			var v = g ? m[g - 1] : 0,
			y = m[g],
			b = {
				left: ("right" == c ? r.right: r.left) - t.rect.left,
				right: ("left" == c ? r.left: r.right) - t.rect.left,
				top: v,
				bottom: y
			};
			return r.left || r.right || (b.bogus = !0),
			e.options.singleCursorHeightPerLine || (b.rtop = h, b.rbottom = f),
			b
		}
		function it(e, t) {
			if (!window.screen || null == screen.logicalXDPI || screen.logicalXDPI == screen.deviceXDPI || !Zr(e)) return t;
			var n = screen.logicalXDPI / screen.deviceXDPI,
			i = screen.logicalYDPI / screen.deviceYDPI;
			return {
				left: t.left * n,
				right: t.right * n,
				top: t.top * i,
				bottom: t.bottom * i
			}
		}
		function rt(e) {
			if (e.measure && (e.measure.cache = {},
			e.measure.heights = null, e.rest)) for (var t = 0; t < e.rest.length; t++) e.measure.caches[t] = {}
		}
		function ot(e) {
			e.display.externalMeasure = null,
			Wr(e.display.lineMeasure);
			for (var t = 0; t < e.display.view.length; t++) rt(e.display.view[t])
		}
		function at(e) {
			ot(e),
			e.display.cachedCharWidth = e.display.cachedTextHeight = e.display.cachedPaddingH = null,
			e.options.lineWrapping || (e.display.maxLineChanged = !0),
			e.display.lineNumChars = null
		}
		function st() {
			return window.pageXOffset || (document.documentElement || document.body).scrollLeft
		}
		function lt() {
			return window.pageYOffset || (document.documentElement || document.body).scrollTop
		}
		function ct(e, t, n, i) {
			if (t.widgets) for (var r = 0; r < t.widgets.length; ++r) if (t.widgets[r].above) {
				var o = _i(t.widgets[r]);
				n.top += o,
				n.bottom += o
			}
			if ("line" == i) return n;
			i || (i = "local");
			var a = er(t);
			if ("local" == i ? a += Fe(e.display) : a -= e.display.viewOffset, "page" == i || "window" == i) {
				var s = e.display.lineSpace.getBoundingClientRect();
				a += s.top + ("window" == i ? 0 : lt());
				var l = s.left + ("window" == i ? 0 : st());
				n.left += l,
				n.right += l
			}
			return n.top += a,
			n.bottom += a,
			n
		}
		function ut(e, t, n) {
			if ("div" == n) return t;
			var i = t.left,
			r = t.top;
			if ("page" == n) i -= st(),
			r -= lt();
			else if ("local" == n || !n) {
				var o = e.display.sizer.getBoundingClientRect();
				i += o.left,
				r += o.top
			}
			var a = e.display.lineSpace.getBoundingClientRect();
			return {
				left: i - a.left,
				top: r - a.top
			}
		}
		function dt(e, t, n, i, r) {
			return i || (i = Vi(e.doc, t.line)),
			ct(e, i, Xe(e, i, t.ch, r), n)
		}
		function ht(e, t, n, i, r, o) {
			function a(t, a) {
				var s = et(e, r, t, a ? "right": "left", o);
				return a ? s.left = s.right: s.right = s.left,
				ct(e, i, s, n)
			}
			function s(e, t) {
				var n = l[t],
				i = n.level % 2;
				return e == Qr(n) && t && n.level < l[t - 1].level ? (n = l[--t], e = Jr(n) - (n.level % 2 ? 0 : 1), i = !0) : e == Jr(n) && t < l.length - 1 && n.level < l[t + 1].level && (n = l[++t], e = Qr(n) - n.level % 2, i = !1),
				i && e == n.to && e > n.from ? a(e - 1) : a(e, i)
			}
			i = i || Vi(e.doc, t.line),
			r || (r = Je(e, i));
			var l = tr(i),
			c = t.ch;
			if (!l) return a(c);
			var u = s(c, ao(l, c));
			return null != ns && (u.other = s(c, ns)),
			u
		}
		function ft(e, t) {
			var n = 0,
			t = pe(e.doc, t);
			e.options.lineWrapping || (n = yt(e.display) * t.ch);
			var i = Vi(e.doc, t.line),
			r = er(i) + Fe(e.display);
			return {
				left: n,
				right: n,
				top: r,
				bottom: r + i.height
			}
		}
		function pt(e, t, n, i) {
			var r = $o(e, t);
			return r.xRel = i,
			n && (r.outside = !0),
			r
		}
		function mt(e, t, n) {
			var i = e.doc;
			if (0 > (n += e.display.viewOffset)) return pt(i.first, 0, !0, -1);
			var r = Ji(i, n),
			o = i.first + i.size - 1;
			if (r > o) return pt(i.first + i.size - 1, Vi(i, o).text.length, !0, 1);
			0 > t && (t = 0);
			for (var a = Vi(i, r);;) {
				var s = gt(e, a, r, t, n),
				l = pi(a),
				c = l && l.find(0, !0);
				if (!l || !(s.ch > c.from.ch || s.ch == c.from.ch && s.xRel > 0)) return s;
				r = Qi(a = c.to.line)
			}
		}
		function gt(e, t, n, i, r) {
			function o(i) {
				var r = ht(e, $o(n, i), "line", t, c);
				return s = !0,
				a > r.bottom ? r.left - l: a < r.top ? r.left + l: (s = !1, r.left)
			}
			var a = r - er(t),
			s = !1,
			l = 2 * e.display.wrapper.clientWidth,
			c = Je(e, t),
			u = tr(t),
			d = t.text.length,
			h = eo(t),
			f = to(t),
			p = o(h),
			m = s,
			g = o(f),
			v = s;
			if (i > g) return pt(n, f, v, 1);
			for (;;) {
				if (u ? f == h || f == lo(t, h, 1) : 1 >= f - h) {
					for (var y = p > i || g - i >= i - p ? h: f, b = i - (y == h ? p: g); Ir(t.text.charAt(y));)++y;
					return pt(n, y, y == h ? m: v, -1 > b ? -1 : b > 1 ? 1 : 0)
				}
				var w = Math.ceil(d / 2),
				k = h + w;
				if (u) {
					k = h;
					for (var x = 0; w > x; ++x) k = lo(t, k, 1)
				}
				var _ = o(k);
				_ > i ? (f = k, g = _, (v = s) && (g += 1e3), d = w) : (h = k, p = _, m = s, d -= w)
			}
		}
		function vt(e) {
			if (null != e.cachedTextHeight) return e.cachedTextHeight;
			if (null == Po) {
				Po = zr("pre");
				for (var t = 0; 49 > t; ++t) Po.appendChild(document.createTextNode("x")),
				Po.appendChild(zr("br"));
				Po.appendChild(document.createTextNode("x"))
			}
			Hr(e.measure, Po);
			var n = Po.offsetHeight / 50;
			return n > 3 && (e.cachedTextHeight = n),
			Wr(e.measure),
			n || 1
		}
		function yt(e) {
			if (null != e.cachedCharWidth) return e.cachedCharWidth;
			var t = zr("span", "xxxxxxxxxx"),
			n = zr("pre", [t]);
			Hr(e.measure, n);
			var i = t.getBoundingClientRect(),
			r = (i.right - i.left) / 10;
			return r > 2 && (e.cachedCharWidth = r),
			r || 10
		}
		function bt(e) {
			e.curOp = {
				cm: e,
				viewChanged: !1,
				startHeight: e.doc.height,
				forceUpdate: !1,
				updateInput: null,
				typing: !1,
				changeObjs: null,
				cursorActivityHandlers: null,
				cursorActivityCalled: 0,
				selectionChanged: !1,
				updateMaxLine: !1,
				scrollLeft: null,
				scrollTop: null,
				scrollToPos: null,
				focus: !1,
				id: ++Fo
			},
			Ho ? Ho.ops.push(e.curOp) : e.curOp.ownsGroup = Ho = {
				ops: [e.curOp],
				delayedCallbacks: []
			}
		}
		function wt(e) {
			var t = e.delayedCallbacks,
			n = 0;
			do {
				for (; n < t.length; n++) t[n].call(null);
				for (var i = 0; i < e.ops.length; i++) {
					var r = e.ops[i];
					if (r.cursorActivityHandlers) for (; r.cursorActivityCalled < r.cursorActivityHandlers.length;) r.cursorActivityHandlers[r.cursorActivityCalled++].call(null, r.cm)
				}
			} while ( n < t . length )
		}
		function kt(e) {
			var t = e.curOp.ownsGroup;
			if (t) try {
				wt(t)
			} finally {
				Ho = null;
				for (var n = 0; n < t.ops.length; n++) t.ops[n].cm.curOp = null;
				xt(t)
			}
		}
		function xt(e) {
			for (var t = e.ops,
			n = 0; n < t.length; n++) _t(t[n]);
			for (n = 0; n < t.length; n++) Ct(t[n]);
			for (n = 0; n < t.length; n++) St(t[n]);
			for (n = 0; n < t.length; n++) Mt(t[n]);
			for (n = 0; n < t.length; n++) Tt(t[n])
		}
		function _t(e) {
			var t = e.cm,
			n = t.display;
			S(t),
			e.updateMaxLine && d(t),
			e.mustUpdate = e.viewChanged || e.forceUpdate || null != e.scrollTop || e.scrollToPos && (e.scrollToPos.from.line < n.viewFrom || e.scrollToPos.to.line >= n.viewTo) || n.maxLineChanged && t.options.lineWrapping,
			e.update = e.mustUpdate && new C(t, e.mustUpdate && {
				top: e.scrollTop,
				ensure: e.scrollToPos
			},
			e.forceUpdate)
		}
		function Ct(e) {
			e.updatedDisplay = e.mustUpdate && M(e.cm, e.update)
		}
		function St(e) {
			var t = e.cm,
			n = t.display;
			e.updatedDisplay && O(t),
			e.barMeasure = f(t),
			n.maxLineChanged && !t.options.lineWrapping && (e.adjustWidthTo = Xe(t, n.maxLine, n.maxLine.text.length).left + 3, t.display.sizerWidth = e.adjustWidthTo, e.barMeasure.scrollWidth = Math.max(n.scroller.clientWidth, n.sizer.offsetLeft + e.adjustWidthTo + Ue(t) + t.display.barWidth), e.maxScrollLeft = Math.max(0, n.sizer.offsetLeft + e.adjustWidthTo - Be(t))),
			(e.updatedDisplay || e.selectionChanged) && (e.preparedSelection = n.input.prepareSelection())
		}
		function Mt(e) {
			var t = e.cm;
			null != e.adjustWidthTo && (t.display.sizer.style.minWidth = e.adjustWidthTo + "px", e.maxScrollLeft < t.doc.scrollLeft && nn(t, Math.min(t.display.scroller.scrollLeft, e.maxScrollLeft), !0), t.display.maxLineChanged = !1),
			e.preparedSelection && t.display.input.showSelection(e.preparedSelection),
			(e.updatedDisplay || e.startHeight != t.doc.height) && v(t, e.barMeasure),
			e.updatedDisplay && L(t, e.barMeasure),
			e.selectionChanged && Pe(t),
			t.state.focused && e.updateInput && t.display.input.reset(e.typing),
			!e.focus || e.focus != Fr() || document.hasFocus && !document.hasFocus() || K(e.cm)
		}
		function Tt(e) {
			var t = e.cm,
			n = t.display,
			i = t.doc;
			if (e.updatedDisplay && T(t, e.update), null == n.wheelStartX || null == e.scrollTop && null == e.scrollLeft && !e.scrollToPos || (n.wheelStartX = n.wheelStartY = null), null == e.scrollTop || n.scroller.scrollTop == e.scrollTop && !e.forceScroll || (i.scrollTop = Math.max(0, Math.min(n.scroller.scrollHeight - n.scroller.clientHeight, e.scrollTop)), n.scrollbars.setScrollTop(i.scrollTop), n.scroller.scrollTop = i.scrollTop), null == e.scrollLeft || n.scroller.scrollLeft == e.scrollLeft && !e.forceScroll || (i.scrollLeft = Math.max(0, Math.min(n.scroller.scrollWidth - n.scroller.clientWidth, e.scrollLeft)), n.scrollbars.setScrollLeft(i.scrollLeft), n.scroller.scrollLeft = i.scrollLeft, w(t)), e.scrollToPos) {
				var r = An(t, pe(i, e.scrollToPos.from), pe(i, e.scrollToPos.to), e.scrollToPos.margin);
				e.scrollToPos.isCursor && t.state.focused && Nn(t, r)
			}
			var o = e.maybeHiddenMarkers,
			a = e.maybeUnhiddenMarkers;
			if (o) for (s = 0; s < o.length; ++s) o[s].lines.length || La(o[s], "hide");
			if (a) for (var s = 0; s < a.length; ++s) a[s].lines.length && La(a[s], "unhide");
			n.wrapper.offsetHeight && (i.scrollTop = t.display.scroller.scrollTop),
			e.changeObjs && La(t, "changes", t, e.changeObjs),
			e.update && e.update.finish()
		}
		function Dt(e, t) {
			if (e.curOp) return t();
			bt(e);
			try {
				return t()
			} finally {
				kt(e)
			}
		}
		function Lt(e, t) {
			return function() {
				if (e.curOp) return t.apply(e, arguments);
				bt(e);
				try {
					return t.apply(e, arguments)
				} finally {
					kt(e)
				}
			}
		}
		function Ot(e) {
			return function() {
				if (this.curOp) return e.apply(this, arguments);
				bt(this);
				try {
					return e.apply(this, arguments)
				} finally {
					kt(this)
				}
			}
		}
		function Nt(e) {
			return function() {
				var t = this.cm;
				if (!t || t.curOp) return e.apply(this, arguments);
				bt(t);
				try {
					return e.apply(this, arguments)
				} finally {
					kt(t)
				}
			}
		}
		function At(e, t, n) {
			this.line = t,
			this.rest = vi(t),
			this.size = this.rest ? Qi(Lr(this.rest)) - n + 1 : 1,
			this.node = this.text = null,
			this.hidden = wi(e, t)
		}
		function Et(e, t, n) {
			for (var i, r = [], o = t; n > o; o = i) {
				var a = new At(e.doc, Vi(e.doc, o), o);
				i = o + a.size,
				r.push(a)
			}
			return r
		}
		function $t(e, t, n, i) {
			null == t && (t = e.doc.first),
			null == n && (n = e.doc.first + e.doc.size),
			i || (i = 0);
			var r = e.display;
			if (i && n < r.viewTo && (null == r.updateLineNumbers || r.updateLineNumbers > t) && (r.updateLineNumbers = t), e.curOp.viewChanged = !0, t >= r.viewTo) Eo && yi(e.doc, t) < r.viewTo && jt(e);
			else if (n <= r.viewFrom) Eo && bi(e.doc, n + i) > r.viewFrom ? jt(e) : (r.viewFrom += i, r.viewTo += i);
			else if (t <= r.viewFrom && n >= r.viewTo) jt(e);
			else if (t <= r.viewFrom)(o = It(e, n, n + i, 1)) ? (r.view = r.view.slice(o.index), r.viewFrom = o.lineN, r.viewTo += i) : jt(e);
			else if (n >= r.viewTo) {
				var o = It(e, t, t, -1);
				o ? (r.view = r.view.slice(0, o.index), r.viewTo = o.lineN) : jt(e)
			} else {
				var a = It(e, t, t, -1),
				s = It(e, n, n + i, 1);
				a && s ? (r.view = r.view.slice(0, a.index).concat(Et(e, a.lineN, s.lineN)).concat(r.view.slice(s.index)), r.viewTo += i) : jt(e)
			}
			var l = r.externalMeasured;
			l && (n < l.lineN ? l.lineN += i: t < l.lineN + l.size && (r.externalMeasured = null))
		}
		function qt(e, t, n) {
			e.curOp.viewChanged = !0;
			var i = e.display,
			r = e.display.externalMeasured;
			if (r && t >= r.lineN && t < r.lineN + r.size && (i.externalMeasured = null), !(t < i.viewFrom || t >= i.viewTo)) {
				var o = i.view[Pt(e, t)];
				if (null != o.node) {
					var a = o.changes || (o.changes = []); - 1 == Or(a, n) && a.push(n)
				}
			}
		}
		function jt(e) {
			e.display.viewFrom = e.display.viewTo = e.doc.first,
			e.display.view = [],
			e.display.viewOffset = 0
		}
		function Pt(e, t) {
			if (t >= e.display.viewTo) return null;
			if (0 > (t -= e.display.viewFrom)) return null;
			for (var n = e.display.view,
			i = 0; i < n.length; i++) if (0 > (t -= n[i].size)) return i
		}
		function It(e, t, n, i) {
			var r, o = Pt(e, t),
			a = e.display.view;
			if (!Eo || n == e.doc.first + e.doc.size) return {
				index: o,
				lineN: n
			};
			for (var s = 0,
			l = e.display.viewFrom; o > s; s++) l += a[s].size;
			if (l != t) {
				if (i > 0) {
					if (o == a.length - 1) return null;
					r = l + a[o].size - t,
					o++
				} else r = l - t;
				t += r,
				n += r
			}
			for (; yi(e.doc, n) != n;) {
				if (o == (0 > i ? 0 : a.length - 1)) return null;
				n += i * a[o - (0 > i ? 1 : 0)].size,
				o += i
			}
			return {
				index: o,
				lineN: n
			}
		}
		function zt(e, t, n) {
			var i = e.display;
			0 == i.view.length || t >= i.viewTo || n <= i.viewFrom ? (i.view = Et(e, t, n), i.viewFrom = t) : (i.viewFrom > t ? i.view = Et(e, t, i.viewFrom).concat(i.view) : i.viewFrom < t && (i.view = i.view.slice(Pt(e, t))), i.viewFrom = t, i.viewTo < n ? i.view = i.view.concat(Et(e, i.viewTo, n)) : i.viewTo > n && (i.view = i.view.slice(0, Pt(e, n)))),
			i.viewTo = n
		}
		function Wt(e) {
			for (var t = e.display.view,
			n = 0,
			i = 0; i < t.length; i++) {
				var r = t[i];
				r.hidden || r.node && !r.changes || ++n
			}
			return n
		}
		function Ht(e) {
			function t() {
				r.activeTouch && (o = setTimeout(function() {
					r.activeTouch = null
				},
				1e3), a = r.activeTouch, a.end = +new Date)
			}
			function n(e) {
				if (1 != e.touches.length) return ! 1;
				var t = e.touches[0];
				return t.radiusX <= 1 && t.radiusY <= 1
			}
			function i(e, t) {
				if (null == t.left) return ! 0;
				var n = t.left - e.left,
				i = t.top - e.top;
				return n * n + i * i > 400
			}
			var r = e.display;
			Ma(r.scroller, "mousedown", Lt(e, Ut)),
			go && 11 > vo ? Ma(r.scroller, "dblclick", Lt(e,
			function(t) {
				if (!_r(e, t)) {
					var n = Rt(e, t);
					if (n && !Zt(e, t) && !Yt(e.display, t)) {
						_a(t);
						var i = e.findWordAt(n);
						be(e.doc, i.anchor, i.head)
					}
				}
			})) : Ma(r.scroller, "dblclick",
			function(t) {
				_r(e, t) || _a(t)
			}),
			No || Ma(r.scroller, "contextmenu",
			function(t) {
				vn(e, t)
			});
			var o, a = {
				end: 0
			};
			Ma(r.scroller, "touchstart",
			function(t) {
				if (!_r(e, t) && !n(t)) {
					clearTimeout(o);
					var i = +new Date;
					r.activeTouch = {
						start: i,
						moved: !1,
						prev: i - a.end <= 300 ? a: null
					},
					1 == t.touches.length && (r.activeTouch.left = t.touches[0].pageX, r.activeTouch.top = t.touches[0].pageY)
				}
			}),
			Ma(r.scroller, "touchmove",
			function() {
				r.activeTouch && (r.activeTouch.moved = !0)
			}),
			Ma(r.scroller, "touchend",
			function(n) {
				var o = r.activeTouch;
				if (o && !Yt(r, n) && null != o.left && !o.moved && new Date - o.start < 300) {
					var a, s = e.coordsChar(r.activeTouch, "page");
					a = !o.prev || i(o, o.prev) ? new ue(s, s) : !o.prev.prev || i(o, o.prev.prev) ? e.findWordAt(s) : new ue($o(s.line, 0), pe(e.doc, $o(s.line + 1, 0))),
					e.setSelection(a.anchor, a.head),
					e.focus(),
					_a(n)
				}
				t()
			}),
			Ma(r.scroller, "touchcancel", t),
			Ma(r.scroller, "scroll",
			function() {
				r.scroller.clientHeight && (tn(e, r.scroller.scrollTop), nn(e, r.scroller.scrollLeft, !0), La(e, "scroll", e))
			}),
			Ma(r.scroller, "mousewheel",
			function(t) {
				rn(e, t)
			}),
			Ma(r.scroller, "DOMMouseScroll",
			function(t) {
				rn(e, t)
			}),
			Ma(r.wrapper, "scroll",
			function() {
				r.wrapper.scrollTop = r.wrapper.scrollLeft = 0
			}),
			r.dragFunctions = {
				enter: function(t) {
					_r(e, t) || Sa(t)
				},
				over: function(t) {
					_r(e, t) || (Jt(e, t), Sa(t))
				},
				start: function(t) {
					Qt(e, t)
				},
				drop: Lt(e, Xt),
				leave: function(t) {
					_r(e, t) || en(e)
				}
			};
			var s = r.input.getField();
			Ma(s, "keyup",
			function(t) {
				hn.call(e, t)
			}),
			Ma(s, "keydown", Lt(e, un)),
			Ma(s, "keypress", Lt(e, fn)),
			Ma(s, "focus", qr(mn, e)),
			Ma(s, "blur", qr(gn, e))
		}
		function Ft(e) {
			var t = e.display; (t.lastWrapHeight != t.wrapper.clientHeight || t.lastWrapWidth != t.wrapper.clientWidth) && (t.cachedCharWidth = t.cachedTextHeight = t.cachedPaddingH = null, t.scrollbarsClipped = !1, e.setSize())
		}
		function Yt(e, t) {
			for (var n = yr(t); n != e.wrapper; n = n.parentNode) if (!n || 1 == n.nodeType && "true" == n.getAttribute("cm-ignore-events") || n.parentNode == e.sizer && n != e.mover) return ! 0
		}
		function Rt(e, t, n, i) {
			var r = e.display;
			if (!n && "true" == yr(t).getAttribute("cm-not-content")) return null;
			var o, a, s = r.lineSpace.getBoundingClientRect();
			try {
				o = t.clientX - s.left,
				a = t.clientY - s.top
			} catch(t) {
				return null
			}
			var l, c = mt(e, o, a);
			if (i && 1 == c.xRel && (l = Vi(e.doc, c.line).text).length == c.ch) {
				var u = ja(l, l.length, e.options.tabSize) - l.length;
				c = $o(c.line, Math.max(0, Math.round((o - Re(e.display).left) / yt(e.display)) - u))
			}
			return c
		}
		function Ut(e) {
			var t = this,
			n = t.display;
			if (! (_r(t, e) || n.activeTouch && n.input.supportsTouch())) {
				if (n.shift = e.shiftKey, Yt(n, e)) return void(yo || (n.scroller.draggable = !1, setTimeout(function() {
					n.scroller.draggable = !0
				},
				100)));
				if (!Zt(t, e)) {
					var i = Rt(t, e);
					switch (window.focus(), br(e)) {
					case 1:
						t.state.selectingText ? t.state.selectingText(e) : i ? Bt(t, e, i) : yr(e) == n.scroller && _a(e);
						break;
					case 2:
						yo && (t.state.lastMiddleDown = +new Date),
						i && be(t.doc, i),
						setTimeout(function() {
							n.input.focus()
						},
						20),
						_a(e);
						break;
					case 3:
						No ? vn(t, e) : pn(t)
					}
				}
			}
		}
		function Bt(e, t, n) {
			go ? setTimeout(qr(K, e), 0) : e.curOp.focus = Fr();
			var i, r = +new Date;
			zo && zo.time > r - 400 && 0 == qo(zo.pos, n) ? i = "triple": Io && Io.time > r - 400 && 0 == qo(Io.pos, n) ? (i = "double", zo = {
				time: r,
				pos: n
			}) : (i = "single", Io = {
				time: r,
				pos: n
			});
			var o, a = e.doc.sel,
			s = To ? t.metaKey: t.ctrlKey;
			e.options.dragDrop && Za && !e.isReadOnly() && "single" == i && (o = a.contains(n)) > -1 && (qo((o = a.ranges[o]).from(), n) < 0 || n.xRel > 0) && (qo(o.to(), n) > 0 || n.xRel < 0) ? Gt(e, t, n, s) : Vt(e, t, n, i, s)
		}
		function Gt(e, t, n, i) {
			var r = e.display,
			o = +new Date,
			a = Lt(e,
			function(s) {
				yo && (r.scroller.draggable = !1),
				e.state.draggingText = !1,
				Da(document, "mouseup", a),
				Da(r.scroller, "drop", a),
				Math.abs(t.clientX - s.clientX) + Math.abs(t.clientY - s.clientY) < 10 && (_a(s), !i && +new Date - 200 < o && be(e.doc, n), yo || go && 9 == vo ? setTimeout(function() {
					document.body.focus(),
					r.input.focus()
				},
				20) : r.input.focus())
			});
			yo && (r.scroller.draggable = !0),
			e.state.draggingText = a,
			r.scroller.dragDrop && r.scroller.dragDrop(),
			Ma(document, "mouseup", a),
			Ma(r.scroller, "drop", a)
		}
		function Vt(e, t, n, i, r) {
			function o(t) {
				if (0 != qo(g, t)) if (g = t, "rect" == i) {
					for (var r = [], o = e.options.tabSize, a = ja(Vi(c, n.line).text, n.ch, o), s = ja(Vi(c, t.line).text, t.ch, o), l = Math.min(a, s), f = Math.max(a, s), p = Math.min(n.line, t.line), m = Math.min(e.lastLine(), Math.max(n.line, t.line)); m >= p; p++) {
						var v = Vi(c, p).text,
						y = Pa(v, l, o);
						l == f ? r.push(new ue($o(p, y), $o(p, y))) : v.length > y && r.push(new ue($o(p, y), $o(p, Pa(v, f, o))))
					}
					r.length || r.push(new ue(n, n)),
					Se(c, de(h.ranges.slice(0, d).concat(r), d), {
						origin: "*mouse",
						scroll: !1
					}),
					e.scrollIntoView(t)
				} else {
					var b = u,
					w = b.anchor,
					k = t;
					if ("single" != i) {
						if ("double" == i) x = e.findWordAt(t);
						else var x = new ue($o(t.line, 0), pe(c, $o(t.line + 1, 0)));
						qo(x.anchor, w) > 0 ? (k = x.head, w = V(b.from(), x.anchor)) : (k = x.anchor, w = G(b.to(), x.head))
					} (r = h.ranges.slice(0))[d] = new ue(pe(c, w), k),
					Se(c, de(r, d), $a)
				}
			}
			function a(t) {
				var n = ++y,
				r = Rt(e, t, !0, "rect" == i);
				if (r) if (0 != qo(r, g)) {
					e.curOp.focus = Fr(),
					o(r);
					var s = b(l, c); (r.line >= s.to || r.line < s.from) && setTimeout(Lt(e,
					function() {
						y == n && a(t)
					}), 150)
				} else {
					var u = t.clientY < v.top ? -20 : t.clientY > v.bottom ? 20 : 0;
					u && setTimeout(Lt(e,
					function() {
						y == n && (l.scroller.scrollTop += u, a(t))
					}), 50)
				}
			}
			function s(t) {
				e.state.selectingText = !1,
				y = 1 / 0,
				_a(t),
				l.input.focus(),
				Da(document, "mousemove", w),
				Da(document, "mouseup", k),
				c.history.lastSelOrigin = null
			}
			var l = e.display,
			c = e.doc;
			_a(t);
			var u, d, h = c.sel,
			f = h.ranges;
			if (r && !t.shiftKey ? (d = c.sel.contains(n), u = d > -1 ? f[d] : new ue(n, n)) : (u = c.sel.primary(), d = c.sel.primIndex), t.altKey) i = "rect",
			r || (u = new ue(n, n)),
			n = Rt(e, t, !0, !0),
			d = -1;
			else if ("double" == i) {
				var p = e.findWordAt(n);
				u = e.display.shift || c.extend ? ye(c, u, p.anchor, p.head) : p
			} else if ("triple" == i) {
				var m = new ue($o(n.line, 0), pe(c, $o(n.line + 1, 0)));
				u = e.display.shift || c.extend ? ye(c, u, m.anchor, m.head) : m
			} else u = ye(c, u, n);
			r ? -1 == d ? (d = f.length, Se(c, de(f.concat([u]), d), {
				scroll: !1,
				origin: "*mouse"
			})) : f.length > 1 && f[d].empty() && "single" == i && !t.shiftKey ? (Se(c, de(f.slice(0, d).concat(f.slice(d + 1)), 0), {
				scroll: !1,
				origin: "*mouse"
			}), h = c.sel) : ke(c, d, u, $a) : (d = 0, Se(c, new ce([u], 0), $a), h = c.sel);
			var g = n,
			v = l.wrapper.getBoundingClientRect(),
			y = 0,
			w = Lt(e,
			function(e) {
				br(e) ? a(e) : s(e)
			}),
			k = Lt(e, s);
			e.state.selectingText = k,
			Ma(document, "mousemove", w),
			Ma(document, "mouseup", k)
		}
		function Kt(e, t, n, i) {
			try {
				var r = t.clientX,
				o = t.clientY
			} catch(t) {
				return ! 1
			}
			if (r >= Math.floor(e.display.gutters.getBoundingClientRect().right)) return ! 1;
			i && _a(t);
			var a = e.display,
			s = a.lineDiv.getBoundingClientRect();
			if (o > s.bottom || !Sr(e, n)) return vr(t);
			o -= s.top - a.viewOffset;
			for (var l = 0; l < e.options.gutters.length; ++l) {
				var c = a.gutters.childNodes[l];
				if (c && c.getBoundingClientRect().right >= r) {
					var u = Ji(e.doc, o),
					d = e.options.gutters[l];
					return La(e, n, e, u, d, t),
					vr(t)
				}
			}
		}
		function Zt(e, t) {
			return Kt(e, t, "gutterClick", !0)
		}
		function Xt(e) {
			var t = this;
			if (en(t), !_r(t, e) && !Yt(t.display, e)) {
				_a(e),
				go && (Yo = +new Date);
				var n = Rt(t, e, !0),
				i = e.dataTransfer.files;
				if (n && !t.isReadOnly()) if (i && i.length && window.FileReader && window.File) for (var r = i.length,
				o = Array(r), a = 0, s = 0; r > s; ++s) !
				function(e, i) {
					if (!t.options.allowDropFileTypes || -1 != Or(t.options.allowDropFileTypes, e.type)) {
						var s = new FileReader;
						s.onload = Lt(t,
						function() {
							var e = s.result;
							if (/[\x00-\x08\x0e-\x1f]{2}/.test(e) && (e = ""), o[i] = e, ++a == r) {
								var l = {
									from: n = pe(t.doc, n),
									to: n,
									text: t.doc.splitLines(o.join(t.doc.lineSeparator())),
									origin: "paste"
								};
								Cn(t.doc, l),
								Ce(t.doc, he(n, Ko(l)))
							}
						}),
						s.readAsText(e)
					}
				} (i[s], s);
				else {
					if (t.state.draggingText && t.doc.sel.contains(n) > -1) return t.state.draggingText(e),
					void setTimeout(function() {
						t.display.input.focus()
					},
					20);
					try {
						if (o = e.dataTransfer.getData("Text")) {
							if (t.state.draggingText && !(To ? e.altKey: e.ctrlKey)) var l = t.listSelections();
							if (Me(t.doc, he(n, n)), l) for (s = 0; s < l.length; ++s) On(t.doc, "", l[s].anchor, l[s].head, "drag");
							t.replaceSelection(o, "around", "paste"),
							t.display.input.focus()
						}
					} catch(e) {}
				}
			}
		}
		function Qt(e, t) {
			if (go && (!e.state.draggingText || +new Date - Yo < 100)) Sa(t);
			else if (!_r(e, t) && !Yt(e.display, t) && (t.dataTransfer.setData("Text", e.getSelection()), t.dataTransfer.setDragImage && !xo)) {
				var n = zr("img", null, null, "position: fixed; left: 0; top: 0;");
				n.src = "data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==",
				ko && (n.width = n.height = 1, e.display.wrapper.appendChild(n), n._top = n.offsetTop),
				t.dataTransfer.setDragImage(n, 0, 0),
				ko && n.parentNode.removeChild(n)
			}
		}
		function Jt(e, t) {
			var n = Rt(e, t);
			if (n) {
				var i = document.createDocumentFragment();
				qe(e, n, i),
				e.display.dragCursor || (e.display.dragCursor = zr("div", null, "CodeMirror-cursors CodeMirror-dragcursors"), e.display.lineSpace.insertBefore(e.display.dragCursor, e.display.cursorDiv)),
				Hr(e.display.dragCursor, i)
			}
		}
		function en(e) {
			e.display.dragCursor && (e.display.lineSpace.removeChild(e.display.dragCursor), e.display.dragCursor = null)
		}
		function tn(e, t) {
			Math.abs(e.doc.scrollTop - t) < 2 || (e.doc.scrollTop = t, fo || D(e, {
				top: t
			}), e.display.scroller.scrollTop != t && (e.display.scroller.scrollTop = t), e.display.scrollbars.setScrollTop(t), fo && D(e), Ie(e, 100))
		}
		function nn(e, t, n) { (n ? t == e.doc.scrollLeft: Math.abs(e.doc.scrollLeft - t) < 2) || (t = Math.min(t, e.display.scroller.scrollWidth - e.display.scroller.clientWidth), e.doc.scrollLeft = t, w(e), e.display.scroller.scrollLeft != t && (e.display.scroller.scrollLeft = t), e.display.scrollbars.setScrollLeft(t))
		}
		function rn(e, t) {
			var n = Bo(t),
			i = n.x,
			r = n.y,
			o = e.display,
			a = o.scroller,
			s = a.scrollWidth > a.clientWidth,
			l = a.scrollHeight > a.clientHeight;
			if (i && s || r && l) {
				if (r && To && yo) e: for (var c = t.target,
				u = o.view; c != a; c = c.parentNode) for (var d = 0; d < u.length; d++) if (u[d].node == c) {
					e.display.currentWheelTarget = c;
					break e
				}
				if (i && !fo && !ko && null != Uo) return r && l && tn(e, Math.max(0, Math.min(a.scrollTop + r * Uo, a.scrollHeight - a.clientHeight))),
				nn(e, Math.max(0, Math.min(a.scrollLeft + i * Uo, a.scrollWidth - a.clientWidth))),
				(!r || r && l) && _a(t),
				void(o.wheelStartX = null);
				if (r && null != Uo) {
					var h = r * Uo,
					f = e.doc.scrollTop,
					p = f + o.wrapper.clientHeight;
					0 > h ? f = Math.max(0, f + h - 50) : p = Math.min(e.doc.height, p + h + 50),
					D(e, {
						top: f,
						bottom: p
					})
				}
				20 > Ro && (null == o.wheelStartX ? (o.wheelStartX = a.scrollLeft, o.wheelStartY = a.scrollTop, o.wheelDX = i, o.wheelDY = r, setTimeout(function() {
					if (null != o.wheelStartX) {
						var e = a.scrollLeft - o.wheelStartX,
						t = a.scrollTop - o.wheelStartY,
						n = t && o.wheelDY && t / o.wheelDY || e && o.wheelDX && e / o.wheelDX;
						o.wheelStartX = o.wheelStartY = null,
						n && (Uo = (Uo * Ro + n) / (Ro + 1), ++Ro)
					}
				},
				200)) : (o.wheelDX += i, o.wheelDY += r))
			}
		}
		function on(e, t, n) {
			if ("string" == typeof t && !(t = aa[t])) return ! 1;
			e.display.input.ensurePolled();
			var i = e.display.shift,
			r = !1;
			try {
				e.isReadOnly() && (e.state.suppressEdits = !0),
				n && (e.display.shift = !1),
				r = t(e) != Aa
			} finally {
				e.display.shift = i,
				e.state.suppressEdits = !1
			}
			return r
		}
		function an(e, t, n) {
			for (var i = 0; i < e.state.keyMaps.length; i++) {
				var r = la(t, e.state.keyMaps[i], n, e);
				if (r) return r
			}
			return e.options.extraKeys && la(t, e.options.extraKeys, n, e) || la(t, e.options.keyMap, n, e)
		}
		function sn(e, t, n, i) {
			var r = e.state.keySeq;
			if (r) {
				if (ca(t)) return "handled";
				Go.set(50,
				function() {
					e.state.keySeq == r && (e.state.keySeq = null, e.display.input.reset())
				}),
				t = r + " " + t
			}
			var o = an(e, t, i);
			return "multi" == o && (e.state.keySeq = t),
			"handled" == o && kr(e, "keyHandled", e, t, n),
			("handled" == o || "multi" == o) && (_a(n), Pe(e)),
			r && !o && /\'$/.test(t) ? (_a(n), !0) : !!o
		}
		function ln(e, t) {
			var n = ua(t, !0);
			return !! n && (t.shiftKey && !e.state.keySeq ? sn(e, "Shift-" + n, t,
			function(t) {
				return on(e, t, !0)
			}) || sn(e, n, t,
			function(t) {
				return ("string" == typeof t ? /^go[A-Z]/.test(t) : t.motion) ? on(e, t) : void 0
			}) : sn(e, n, t,
			function(t) {
				return on(e, t)
			}))
		}
		function cn(e, t, n) {
			return sn(e, "'" + n + "'", t,
			function(t) {
				return on(e, t, !0)
			})
		}
		function un(e) {
			var t = this;
			if (t.curOp.focus = Fr(), !_r(t, e)) {
				go && 11 > vo && 27 == e.keyCode && (e.returnValue = !1);
				var n = e.keyCode;
				t.display.shift = 16 == n || e.shiftKey;
				var i = ln(t, e);
				ko && (Vo = i ? n: null, !i && 88 == n && !Ja && (To ? e.metaKey: e.ctrlKey) && t.replaceSelection("", null, "cut")),
				18 != n || /\bCodeMirror-crosshair\b/.test(t.display.lineDiv.className) || dn(t)
			}
		}
		function dn(e) {
			function t(e) {
				18 != e.keyCode && e.altKey || (Ga(n, "CodeMirror-crosshair"), Da(document, "keyup", t), Da(document, "mouseover", t))
			}
			var n = e.display.lineDiv;
			Va(n, "CodeMirror-crosshair"),
			Ma(document, "keyup", t),
			Ma(document, "mouseover", t)
		}
		function hn(e) {
			16 == e.keyCode && (this.doc.sel.shift = !1),
			_r(this, e)
		}
		function fn(e) {
			var t = this;
			if (! (Yt(t.display, e) || _r(t, e) || e.ctrlKey && !e.altKey || To && e.metaKey)) {
				var n = e.keyCode,
				i = e.charCode;
				if (ko && n == Vo) return Vo = null,
				void _a(e);
				ko && (!e.which || e.which < 10) && ln(t, e) || cn(t, e, String.fromCharCode(null == i ? n: i)) || t.display.input.onKeyPress(e)
			}
		}
		function pn(e) {
			e.state.delayingBlurEvent = !0,
			setTimeout(function() {
				e.state.delayingBlurEvent && (e.state.delayingBlurEvent = !1, gn(e))
			},
			100)
		}
		function mn(e) {
			e.state.delayingBlurEvent && (e.state.delayingBlurEvent = !1),
			"nocursor" != e.options.readOnly && (e.state.focused || (La(e, "focus", e), e.state.focused = !0, Va(e.display.wrapper, "CodeMirror-focused"), e.curOp || e.display.selForContextMenu == e.doc.sel || (e.display.input.reset(), yo && setTimeout(function() {
				e.display.input.reset(!0)
			},
			20)), e.display.input.receivedFocus()), Pe(e))
		}
		function gn(e) {
			e.state.delayingBlurEvent || (e.state.focused && (La(e, "blur", e), e.state.focused = !1, Ga(e.display.wrapper, "CodeMirror-focused")), clearInterval(e.display.blinker), setTimeout(function() {
				e.state.focused || (e.display.shift = !1)
			},
			150))
		}
		function vn(e, t) {
			Yt(e.display, t) || yn(e, t) || _r(e, t, "contextmenu") || e.display.input.onContextMenu(t)
		}
		function yn(e, t) {
			return !! Sr(e, "gutterContextMenu") && Kt(e, t, "gutterContextMenu", !1)
		}
		function bn(e, t) {
			if (qo(e, t.from) < 0) return e;
			if (qo(e, t.to) <= 0) return Ko(t);
			var n = e.line + t.text.length - (t.to.line - t.from.line) - 1,
			i = e.ch;
			return e.line == t.to.line && (i += Ko(t).ch - t.to.ch),
			$o(n, i)
		}
		function wn(e, t) {
			for (var n = [], i = 0; i < e.sel.ranges.length; i++) {
				var r = e.sel.ranges[i];
				n.push(new ue(bn(r.anchor, t), bn(r.head, t)))
			}
			return de(n, e.sel.primIndex)
		}
		function kn(e, t, n) {
			return e.line == t.line ? $o(n.line, e.ch - t.ch + n.ch) : $o(n.line + (e.line - t.line), e.ch)
		}
		function xn(e, t, n) {
			for (var i = [], r = $o(e.first, 0), o = r, a = 0; a < t.length; a++) {
				var s = t[a],
				l = kn(s.from, r, o),
				c = kn(Ko(s), r, o);
				if (r = s.to, o = c, "around" == n) {
					var u = e.sel.ranges[a],
					d = qo(u.head, u.anchor) < 0;
					i[a] = new ue(d ? c: l, d ? l: c)
				} else i[a] = new ue(l, l)
			}
			return new ce(i, e.sel.primIndex)
		}
		function _n(e, t, n) {
			var i = {
				canceled: !1,
				from: t.from,
				to: t.to,
				text: t.text,
				origin: t.origin,
				cancel: function() {
					this.canceled = !0
				}
			};
			return n && (i.update = function(t, n, i, r) {
				t && (this.from = pe(e, t)),
				n && (this.to = pe(e, n)),
				i && (this.text = i),
				void 0 !== r && (this.origin = r)
			}),
			La(e, "beforeChange", e, i),
			e.cm && La(e.cm, "beforeChange", e.cm, i),
			i.canceled ? null: {
				from: i.from,
				to: i.to,
				text: i.text,
				origin: i.origin
			}
		}
		function Cn(e, t, n) {
			if (e.cm) {
				if (!e.cm.curOp) return Lt(e.cm, Cn)(e, t, n);
				if (e.cm.state.suppressEdits) return
			}
			if (! (Sr(e, "beforeChange") || e.cm && Sr(e.cm, "beforeChange")) || (t = _n(e, t, !0))) {
				var i = Ao && !n && ai(e, t.from, t.to);
				if (i) for (var r = i.length - 1; r >= 0; --r) Sn(e, {
					from: i[r].from,
					to: i[r].to,
					text: r ? [""] : t.text
				});
				else Sn(e, t)
			}
		}
		function Sn(e, t) {
			if (1 != t.text.length || "" != t.text[0] || 0 != qo(t.from, t.to)) {
				var n = wn(e, t);
				ar(e, t, n, e.cm ? e.cm.curOp.id: NaN),
				Dn(e, t, n, ii(e, t));
				var i = [];
				Bi(e,
				function(e, n) {
					n || -1 != Or(i, e.history) || (gr(e.history, t), i.push(e.history)),
					Dn(e, t, null, ii(e, t))
				})
			}
		}
		function Mn(e, t, n) {
			if (!e.cm || !e.cm.state.suppressEdits) {
				for (var i, r = e.history,
				o = e.sel,
				a = "undo" == t ? r.done: r.undone, s = "undo" == t ? r.undone: r.done, l = 0; l < a.length && (i = a[l], n ? !i.ranges || i.equals(e.sel) : i.ranges); l++);
				if (l != a.length) {
					for (r.lastOrigin = r.lastSelOrigin = null; (i = a.pop()).ranges;) {
						if (cr(i, s), n && !i.equals(e.sel)) return void Se(e, i, {
							clearRedo: !1
						});
						o = i
					}
					var c = [];
					cr(o, s),
					s.push({
						changes: c,
						generation: r.generation
					}),
					r.generation = i.generation || ++r.maxGeneration;
					for (var u = Sr(e, "beforeChange") || e.cm && Sr(e.cm, "beforeChange"), l = i.changes.length - 1; l >= 0; --l) {
						var d = i.changes[l];
						if (d.origin = t, u && !_n(e, d, !1)) return void(a.length = 0);
						c.push(ir(e, d));
						var h = l ? wn(e, d) : Lr(a);
						Dn(e, d, h, oi(e, d)),
						!l && e.cm && e.cm.scrollIntoView({
							from: d.from,
							to: Ko(d)
						});
						var f = [];
						Bi(e,
						function(e, t) {
							t || -1 != Or(f, e.history) || (gr(e.history, d), f.push(e.history)),
							Dn(e, d, null, oi(e, d))
						})
					}
				}
			}
		}
		function Tn(e, t) {
			if (0 != t && (e.first += t, e.sel = new ce(Nr(e.sel.ranges,
			function(e) {
				return new ue($o(e.anchor.line + t, e.anchor.ch), $o(e.head.line + t, e.head.ch))
			}), e.sel.primIndex), e.cm)) {
				$t(e.cm, e.first, e.first - t, t);
				for (var n = e.cm.display,
				i = n.viewFrom; i < n.viewTo; i++) qt(e.cm, i, "gutter")
			}
		}
		function Dn(e, t, n, i) {
			if (e.cm && !e.cm.curOp) return Lt(e.cm, Dn)(e, t, n, i);
			if (t.to.line < e.first) Tn(e, t.text.length - 1 - (t.to.line - t.from.line));
			else if (! (t.from.line > e.lastLine())) {
				if (t.from.line < e.first) {
					var r = t.text.length - 1 - (e.first - t.from.line);
					Tn(e, r),
					t = {
						from: $o(e.first, 0),
						to: $o(t.to.line + r, t.to.ch),
						text: [Lr(t.text)],
						origin: t.origin
					}
				}
				var o = e.lastLine();
				t.to.line > o && (t = {
					from: t.from,
					to: $o(o, Vi(e, o).text.length),
					text: [t.text[0]],
					origin: t.origin
				}),
				t.removed = Ki(e, t.from, t.to),
				n || (n = wn(e, t)),
				e.cm ? Ln(e.cm, t, i) : Yi(e, t, i),
				Me(e, n, Ea)
			}
		}
		function Ln(e, t, n) {
			var i = e.doc,
			o = e.display,
			a = t.from,
			s = t.to,
			l = !1,
			c = a.line;
			e.options.lineWrapping || (c = Qi(gi(Vi(i, a.line))), i.iter(c, s.line + 1,
			function(e) {
				return e == o.maxLine ? (l = !0, !0) : void 0
			})),
			i.sel.contains(t.from, t.to) > -1 && Cr(e),
			Yi(i, t, n, r(e)),
			e.options.lineWrapping || (i.iter(c, a.line + t.text.length,
			function(e) {
				var t = u(e);
				t > o.maxLineLength && (o.maxLine = e, o.maxLineLength = t, o.maxLineChanged = !0, l = !1)
			}), l && (e.curOp.updateMaxLine = !0)),
			i.frontier = Math.min(i.frontier, a.line),
			Ie(e, 400);
			var d = t.text.length - (s.line - a.line) - 1;
			t.full ? $t(e) : a.line != s.line || 1 != t.text.length || Fi(e.doc, t) ? $t(e, a.line, s.line + 1, d) : qt(e, a.line, "text");
			var h = Sr(e, "changes"),
			f = Sr(e, "change");
			if (f || h) {
				var p = {
					from: a,
					to: s,
					text: t.text,
					removed: t.removed,
					origin: t.origin
				};
				f && kr(e, "change", e, p),
				h && (e.curOp.changeObjs || (e.curOp.changeObjs = [])).push(p)
			}
			e.display.selForContextMenu = null
		}
		function On(e, t, n, i, r) {
			if (i || (i = n), qo(i, n) < 0) {
				var o = i;
				i = n,
				n = o
			}
			"string" == typeof t && (t = e.splitLines(t)),
			Cn(e, {
				from: n,
				to: i,
				text: t,
				origin: r
			})
		}
		function Nn(e, t) {
			if (!_r(e, "scrollCursorIntoView")) {
				var n = e.display,
				i = n.sizer.getBoundingClientRect(),
				r = null;
				if (t.top + i.top < 0 ? r = !0 : t.bottom + i.top > (window.innerHeight || document.documentElement.clientHeight) && (r = !1), null != r && !Co) {
					var o = zr("div", "​", null, "position: absolute; top: " + (t.top - n.viewOffset - Fe(e.display)) + "px; height: " + (t.bottom - t.top + Ue(e) + n.barHeight) + "px; left: " + t.left + "px; width: 2px;");
					e.display.lineSpace.appendChild(o),
					o.scrollIntoView(r),
					e.display.lineSpace.removeChild(o)
				}
			}
		}
		function An(e, t, n, i) {
			null == i && (i = 0);
			for (var r = 0; 5 > r; r++) {
				var o = !1,
				a = ht(e, t),
				s = n && n != t ? ht(e, n) : a,
				l = $n(e, Math.min(a.left, s.left), Math.min(a.top, s.top) - i, Math.max(a.left, s.left), Math.max(a.bottom, s.bottom) + i),
				c = e.doc.scrollTop,
				u = e.doc.scrollLeft;
				if (null != l.scrollTop && (tn(e, l.scrollTop), Math.abs(e.doc.scrollTop - c) > 1 && (o = !0)), null != l.scrollLeft && (nn(e, l.scrollLeft), Math.abs(e.doc.scrollLeft - u) > 1 && (o = !0)), !o) break
			}
			return a
		}
		function En(e, t, n, i, r) {
			var o = $n(e, t, n, i, r);
			null != o.scrollTop && tn(e, o.scrollTop),
			null != o.scrollLeft && nn(e, o.scrollLeft)
		}
		function $n(e, t, n, i, r) {
			var o = e.display,
			a = vt(e.display);
			0 > n && (n = 0);
			var s = e.curOp && null != e.curOp.scrollTop ? e.curOp.scrollTop: o.scroller.scrollTop,
			l = Ge(e),
			c = {};
			r - n > l && (r = n + l);
			var u = e.doc.height + Ye(o),
			d = a > n,
			h = r > u - a;
			if (s > n) c.scrollTop = d ? 0 : n;
			else if (r > s + l) {
				var f = Math.min(n, (h ? u: r) - l);
				f != s && (c.scrollTop = f)
			}
			var p = e.curOp && null != e.curOp.scrollLeft ? e.curOp.scrollLeft: o.scroller.scrollLeft,
			m = Be(e) - (e.options.fixedGutter ? o.gutters.offsetWidth: 0),
			g = i - t > m;
			return g && (i = t + m),
			10 > t ? c.scrollLeft = 0 : p > t ? c.scrollLeft = Math.max(0, t - (g ? 0 : 10)) : i > m + p - 3 && (c.scrollLeft = i + (g ? 0 : 10) - m),
			c
		}
		function qn(e, t, n) { (null != t || null != n) && Pn(e),
			null != t && (e.curOp.scrollLeft = (null == e.curOp.scrollLeft ? e.doc.scrollLeft: e.curOp.scrollLeft) + t),
			null != n && (e.curOp.scrollTop = (null == e.curOp.scrollTop ? e.doc.scrollTop: e.curOp.scrollTop) + n)
		}
		function jn(e) {
			Pn(e);
			var t = e.getCursor(),
			n = t,
			i = t;
			e.options.lineWrapping || (n = t.ch ? $o(t.line, t.ch - 1) : t, i = $o(t.line, t.ch + 1)),
			e.curOp.scrollToPos = {
				from: n,
				to: i,
				margin: e.options.cursorScrollMargin,
				isCursor: !0
			}
		}
		function Pn(e) {
			var t = e.curOp.scrollToPos;
			if (t) {
				e.curOp.scrollToPos = null;
				var n = ft(e, t.from),
				i = ft(e, t.to),
				r = $n(e, Math.min(n.left, i.left), Math.min(n.top, i.top) - t.margin, Math.max(n.right, i.right), Math.max(n.bottom, i.bottom) + t.margin);
				e.scrollTo(r.scrollLeft, r.scrollTop)
			}
		}
		function In(e, t, n, i) {
			var r, o = e.doc;
			null == n && (n = "add"),
			"smart" == n && (o.mode.indent ? r = He(e, t) : n = "prev");
			var a = e.options.tabSize,
			s = Vi(o, t),
			l = ja(s.text, null, a);
			s.stateAfter && (s.stateAfter = null);
			var c, u = s.text.match(/^\s*/)[0];
			if (i || /\S/.test(s.text)) {
				if ("smart" == n && ((c = o.mode.indent(r, s.text.slice(u.length), s.text)) == Aa || c > 150)) {
					if (!i) return;
					n = "prev"
				}
			} else c = 0,
			n = "not";
			"prev" == n ? c = t > o.first ? ja(Vi(o, t - 1).text, null, a) : 0 : "add" == n ? c = l + e.options.indentUnit: "subtract" == n ? c = l - e.options.indentUnit: "number" == typeof n && (c = l + n),
			c = Math.max(0, c);
			var d = "",
			h = 0;
			if (e.options.indentWithTabs) for (f = Math.floor(c / a); f; --f) h += a,
			d += "\t";
			if (c > h && (d += Dr(c - h)), d != u) return On(o, d, $o(t, 0), $o(t, u.length), "+input"),
			s.stateAfter = null,
			!0;
			for (var f = 0; f < o.sel.ranges.length; f++) {
				var p = o.sel.ranges[f];
				if (p.head.line == t && p.head.ch < u.length) {
					ke(o, f, new ue(h = $o(t, u.length), h));
					break
				}
			}
		}
		function zn(e, t, n, i) {
			var r = t,
			o = t;
			return "number" == typeof t ? o = Vi(e, fe(e, t)) : r = Qi(t),
			null == r ? null: (i(o, r) && e.cm && qt(e.cm, r, n), o)
		}
		function Wn(e, t) {
			for (var n = e.doc.sel.ranges,
			i = [], r = 0; r < n.length; r++) {
				for (var o = t(n[r]); i.length && qo(o.from, Lr(i).to) <= 0;) {
					var a = i.pop();
					if (qo(a.from, o.from) < 0) {
						o.from = a.from;
						break
					}
				}
				i.push(o)
			}
			Dt(e,
			function() {
				for (var t = i.length - 1; t >= 0; t--) On(e.doc, "", i[t].from, i[t].to, "+delete");
				jn(e)
			})
		}
		function Hn(e, t, n, i, r) {
			function o() {
				var t = s + n;
				return ! (t < e.first || t >= e.first + e.size) && (s = t, u = Vi(e, t))
			}
			function a(e) {
				var t = (r ? lo: co)(u, l, n, !0);
				if (null == t) {
					if (e || !o()) return ! 1;
					l = r ? (0 > n ? to: eo)(u) : 0 > n ? u.text.length: 0
				} else l = t;
				return ! 0
			}
			var s = t.line,
			l = t.ch,
			c = n,
			u = Vi(e, s);
			if ("char" == i) a();
			else if ("column" == i) a(!0);
			else if ("word" == i || "group" == i) for (var d = null,
			h = "group" == i,
			f = e.cm && e.cm.getHelper(t, "wordChars"), p = !0; ! (0 > n) || a(!p); p = !1) {
				var m = u.text.charAt(l) || "\n",
				g = jr(m, f) ? "w": h && "\n" == m ? "n": !h || /\s/.test(m) ? null: "p";
				if (!h || p || g || (g = "s"), d && d != g) {
					0 > n && (n = 1, a());
					break
				}
				if (g && (d = g), n > 0 && !a(!p)) break
			}
			var v = Ne(e, $o(s, l), t, c, !0);
			return qo(t, v) || (v.hitSide = !0),
			v
		}
		function Fn(e, t, n, i) {
			var r, o = e.doc,
			a = t.left;
			if ("page" == i) {
				var s = Math.min(e.display.wrapper.clientHeight, window.innerHeight || document.documentElement.clientHeight);
				r = t.top + n * (s - (0 > n ? 1.5 : .5) * vt(e.display))
			} else "line" == i && (r = n > 0 ? t.bottom + 3 : t.top - 3);
			for (;;) {
				var l = mt(e, a, r);
				if (!l.outside) break;
				if (0 > n ? 0 >= r: r >= o.height) {
					l.hitSide = !0;
					break
				}
				r += 5 * n
			}
			return l
		}
		function Yn(t, n, i, r) {
			e.defaults[t] = n,
			i && (Xo[t] = r ?
			function(e, t, n) {
				n != Qo && i(e, t, n)
			}: i)
		}
		function Rn(e) {
			for (var t, n, i, r, o = e.split(/-(?!$)/), e = o[o.length - 1], a = 0; a < o.length - 1; a++) {
				var s = o[a];
				if (/^(cmd|meta|m)$/i.test(s)) r = !0;
				else if (/^a(lt)?$/i.test(s)) t = !0;
				else if (/^(c|ctrl|control)$/i.test(s)) n = !0;
				else {
					if (!/^s(hift)$/i.test(s)) throw new Error("Unrecognized modifier name: " + s);
					i = !0
				}
			}
			return t && (e = "Alt-" + e),
			n && (e = "Ctrl-" + e),
			r && (e = "Cmd-" + e),
			i && (e = "Shift-" + e),
			e
		}
		function Un(e) {
			return "string" == typeof e ? sa[e] : e
		}
		function Bn(e, t, n, i, r) {
			if (i && i.shared) return Gn(e, t, n, i, r);
			if (e.cm && !e.cm.curOp) return Lt(e.cm, Bn)(e, t, n, i, r);
			var o = new fa(e, r),
			a = qo(t, n);
			if (i && $r(i, o, !1), a > 0 || 0 == a && !1 !== o.clearWhenEmpty) return o;
			if (o.replacedWith && (o.collapsed = !0, o.widgetNode = zr("span", [o.replacedWith], "CodeMirror-widget"), i.handleMouseEvents || o.widgetNode.setAttribute("cm-ignore-events", "true"), i.insertLeft && (o.widgetNode.insertLeft = !0)), o.collapsed) {
				if (mi(e, t.line, t, n, o) || t.line != n.line && mi(e, n.line, t, n, o)) throw new Error("Inserting collapsed marker partially overlapping an existing one");
				Eo = !0
			}
			o.addToHistory && ar(e, {
				from: t,
				to: n,
				origin: "markText"
			},
			e.sel, NaN);
			var s, l = t.line,
			c = e.cm;
			if (e.iter(l, n.line + 1,
			function(e) {
				c && o.collapsed && !c.options.lineWrapping && gi(e) == c.display.maxLine && (s = !0),
				o.collapsed && l != t.line && Xi(e, 0),
				ei(e, new Xn(o, l == t.line ? t.ch: null, l == n.line ? n.ch: null)),
				++l
			}), o.collapsed && e.iter(t.line, n.line + 1,
			function(t) {
				wi(e, t) && Xi(t, 0)
			}), o.clearOnEnter && Ma(o, "beforeCursorEnter",
			function() {
				o.clear()
			}), o.readOnly && (Ao = !0, (e.history.done.length || e.history.undone.length) && e.clearHistory()), o.collapsed && (o.id = ++ha, o.atomic = !0), c) {
				if (s && (c.curOp.updateMaxLine = !0), o.collapsed) $t(c, t.line, n.line + 1);
				else if (o.className || o.title || o.startStyle || o.endStyle || o.css) for (var u = t.line; u <= n.line; u++) qt(c, u, "text");
				o.atomic && De(c.doc),
				kr(c, "markerAdded", c, o)
			}
			return o
		}
		function Gn(e, t, n, i, r) { (i = $r(i)).shared = !1;
			var o = [Bn(e, t, n, i, r)],
			a = o[0],
			s = i.widgetNode;
			return Bi(e,
			function(e) {
				s && (i.widgetNode = s.cloneNode(!0)),
				o.push(Bn(e, pe(e, t), pe(e, n), i, r));
				for (var l = 0; l < e.linked.length; ++l) if (e.linked[l].isParent) return;
				a = Lr(o)
			}),
			new pa(o, a)
		}
		function Vn(e) {
			return e.findMarks($o(e.first, 0), e.clipPos($o(e.lastLine())),
			function(e) {
				return e.parent
			})
		}
		function Kn(e, t) {
			for (var n = 0; n < t.length; n++) {
				var i = t[n],
				r = i.find(),
				o = e.clipPos(r.from),
				a = e.clipPos(r.to);
				if (qo(o, a)) {
					var s = Bn(e, o, a, i.primary, i.primary.type);
					i.markers.push(s),
					s.parent = i
				}
			}
		}
		function Zn(e) {
			for (var t = 0; t < e.length; t++) {
				var n = e[t],
				i = [n.primary.doc];
				Bi(n.primary.doc,
				function(e) {
					i.push(e)
				});
				for (var r = 0; r < n.markers.length; r++) {
					var o = n.markers[r]; - 1 == Or(i, o.doc) && (o.parent = null, n.markers.splice(r--, 1))
				}
			}
		}
		function Xn(e, t, n) {
			this.marker = e,
			this.from = t,
			this.to = n
		}
		function Qn(e, t) {
			if (e) for (var n = 0; n < e.length; ++n) {
				var i = e[n];
				if (i.marker == t) return i
			}
		}
		function Jn(e, t) {
			for (var n, i = 0; i < e.length; ++i) e[i] != t && (n || (n = [])).push(e[i]);
			return n
		}
		function ei(e, t) {
			e.markedSpans = e.markedSpans ? e.markedSpans.concat([t]) : [t],
			t.marker.attachLine(e)
		}
		function ti(e, t, n) {
			if (e) for (var i, r = 0; r < e.length; ++r) {
				var o = e[r],
				a = o.marker;
				if (null == o.from || (a.inclusiveLeft ? o.from <= t: o.from < t) || o.from == t && "bookmark" == a.type && (!n || !o.marker.insertLeft)) {
					var s = null == o.to || (a.inclusiveRight ? o.to >= t: o.to > t); (i || (i = [])).push(new Xn(a, o.from, s ? null: o.to))
				}
			}
			return i
		}
		function ni(e, t, n) {
			if (e) for (var i, r = 0; r < e.length; ++r) {
				var o = e[r],
				a = o.marker;
				if (null == o.to || (a.inclusiveRight ? o.to >= t: o.to > t) || o.from == t && "bookmark" == a.type && (!n || o.marker.insertLeft)) {
					var s = null == o.from || (a.inclusiveLeft ? o.from <= t: o.from < t); (i || (i = [])).push(new Xn(a, s ? null: o.from - t, null == o.to ? null: o.to - t))
				}
			}
			return i
		}
		function ii(e, t) {
			if (t.full) return null;
			var n = ge(e, t.from.line) && Vi(e, t.from.line).markedSpans,
			i = ge(e, t.to.line) && Vi(e, t.to.line).markedSpans;
			if (!n && !i) return null;
			var r = t.from.ch,
			o = t.to.ch,
			a = 0 == qo(t.from, t.to),
			s = ti(n, r, a),
			l = ni(i, o, a),
			c = 1 == t.text.length,
			u = Lr(t.text).length + (c ? r: 0);
			if (s) for (g = 0; g < s.length; ++g) null == (d = s[g]).to && ((h = Qn(l, d.marker)) ? c && (d.to = null == h.to ? null: h.to + u) : d.to = r);
			if (l) for (g = 0; g < l.length; ++g) {
				var d = l[g];
				if (null != d.to && (d.to += u), null == d.from) {
					var h = Qn(s, d.marker);
					h || (d.from = u, c && (s || (s = [])).push(d))
				} else d.from += u,
				c && (s || (s = [])).push(d)
			}
			s && (s = ri(s)),
			l && l != s && (l = ri(l));
			var f = [s];
			if (!c) {
				var p, m = t.text.length - 2;
				if (m > 0 && s) for (g = 0; g < s.length; ++g) null == s[g].to && (p || (p = [])).push(new Xn(s[g].marker, null, null));
				for (var g = 0; m > g; ++g) f.push(p);
				f.push(l)
			}
			return f
		}
		function ri(e) {
			for (var t = 0; t < e.length; ++t) {
				var n = e[t];
				null != n.from && n.from == n.to && !1 !== n.marker.clearWhenEmpty && e.splice(t--, 1)
			}
			return e.length ? e: null
		}
		function oi(e, t) {
			var n = hr(e, t),
			i = ii(e, t);
			if (!n) return i;
			if (!i) return n;
			for (var r = 0; r < n.length; ++r) {
				var o = n[r],
				a = i[r];
				if (o && a) e: for (var s = 0; s < a.length; ++s) {
					for (var l = a[s], c = 0; c < o.length; ++c) if (o[c].marker == l.marker) continue e;
					o.push(l)
				} else a && (n[r] = a)
			}
			return n
		}
		function ai(e, t, n) {
			var i = null;
			if (e.iter(t.line, n.line + 1,
			function(e) {
				if (e.markedSpans) for (var t = 0; t < e.markedSpans.length; ++t) {
					var n = e.markedSpans[t].marker; ! n.readOnly || i && -1 != Or(i, n) || (i || (i = [])).push(n)
				}
			}), !i) return null;
			for (var r = [{
				from: t,
				to: n
			}], o = 0; o < i.length; ++o) for (var a = i[o], s = a.find(0), l = 0; l < r.length; ++l) {
				var c = r[l];
				if (! (qo(c.to, s.from) < 0 || qo(c.from, s.to) > 0)) {
					var u = [l, 1],
					d = qo(c.from, s.from),
					h = qo(c.to, s.to); (0 > d || !a.inclusiveLeft && !d) && u.push({
						from: c.from,
						to: s.from
					}),
					(h > 0 || !a.inclusiveRight && !h) && u.push({
						from: s.to,
						to: c.to
					}),
					r.splice.apply(r, u),
					l += u.length - 1
				}
			}
			return r
		}
		function si(e) {
			var t = e.markedSpans;
			if (t) {
				for (var n = 0; n < t.length; ++n) t[n].marker.detachLine(e);
				e.markedSpans = null
			}
		}
		function li(e, t) {
			if (t) {
				for (var n = 0; n < t.length; ++n) t[n].marker.attachLine(e);
				e.markedSpans = t
			}
		}
		function ci(e) {
			return e.inclusiveLeft ? -1 : 0
		}
		function ui(e) {
			return e.inclusiveRight ? 1 : 0
		}
		function di(e, t) {
			var n = e.lines.length - t.lines.length;
			if (0 != n) return n;
			var i = e.find(),
			r = t.find(),
			o = qo(i.from, r.from) || ci(e) - ci(t);
			if (o) return - o;
			var a = qo(i.to, r.to) || ui(e) - ui(t);
			return a || t.id - e.id
		}
		function hi(e, t) {
			var n, i = Eo && e.markedSpans;
			if (i) for (var r, o = 0; o < i.length; ++o)(r = i[o]).marker.collapsed && null == (t ? r.from: r.to) && (!n || di(n, r.marker) < 0) && (n = r.marker);
			return n
		}
		function fi(e) {
			return hi(e, !0)
		}
		function pi(e) {
			return hi(e, !1)
		}
		function mi(e, t, n, i, r) {
			var o = Vi(e, t),
			a = Eo && o.markedSpans;
			if (a) for (var s = 0; s < a.length; ++s) {
				var l = a[s];
				if (l.marker.collapsed) {
					var c = l.marker.find(0),
					u = qo(c.from, n) || ci(l.marker) - ci(r),
					d = qo(c.to, i) || ui(l.marker) - ui(r);
					if (! (u >= 0 && 0 >= d || 0 >= u && d >= 0) && (0 >= u && (qo(c.to, n) > 0 || l.marker.inclusiveRight && r.inclusiveLeft) || u >= 0 && (qo(c.from, i) < 0 || l.marker.inclusiveLeft && r.inclusiveRight))) return ! 0
				}
			}
		}
		function gi(e) {
			for (var t; t = fi(e);) e = t.find( - 1, !0).line;
			return e
		}
		function vi(e) {
			for (var t, n; t = pi(e);) e = t.find(1, !0).line,
			(n || (n = [])).push(e);
			return n
		}
		function yi(e, t) {
			var n = Vi(e, t),
			i = gi(n);
			return n == i ? t: Qi(i)
		}
		function bi(e, t) {
			if (t > e.lastLine()) return t;
			var n, i = Vi(e, t);
			if (!wi(e, i)) return t;
			for (; n = pi(i);) i = n.find(1, !0).line;
			return Qi(i) + 1
		}
		function wi(e, t) {
			var n = Eo && t.markedSpans;
			if (n) for (var i, r = 0; r < n.length; ++r) if ((i = n[r]).marker.collapsed) {
				if (null == i.from) return ! 0;
				if (!i.marker.widgetNode && 0 == i.from && i.marker.inclusiveLeft && ki(e, t, i)) return ! 0
			}
		}
		function ki(e, t, n) {
			if (null == n.to) {
				var i = n.marker.find(1, !0);
				return ki(e, i.line, Qn(i.line.markedSpans, n.marker))
			}
			if (n.marker.inclusiveRight && n.to == t.text.length) return ! 0;
			for (var r, o = 0; o < t.markedSpans.length; ++o) if ((r = t.markedSpans[o]).marker.collapsed && !r.marker.widgetNode && r.from == n.to && (null == r.to || r.to != n.from) && (r.marker.inclusiveLeft || n.marker.inclusiveRight) && ki(e, t, r)) return ! 0
		}
		function xi(e, t, n) {
			er(t) < (e.curOp && e.curOp.scrollTop || e.doc.scrollTop) && qn(e, null, n)
		}
		function _i(e) {
			if (null != e.height) return e.height;
			var t = e.doc.cm;
			if (!t) return 0;
			if (!Ra(document.body, e.node)) {
				var n = "position: relative;";
				e.coverGutter && (n += "margin-left: -" + t.display.gutters.offsetWidth + "px;"),
				e.noHScroll && (n += "width: " + t.display.wrapper.clientWidth + "px;"),
				Hr(t.display.measure, zr("div", [e.node], null, n))
			}
			return e.height = e.node.parentNode.offsetHeight
		}
		function Ci(e, t, n, i) {
			var r = new ma(e, n, i),
			o = e.cm;
			return o && r.noHScroll && (o.display.alignWidgets = !0),
			zn(e, t, "widget",
			function(t) {
				var n = t.widgets || (t.widgets = []);
				if (null == r.insertAt ? n.push(r) : n.splice(Math.min(n.length - 1, Math.max(0, r.insertAt)), 0, r), r.line = t, o && !wi(e, t)) {
					var i = er(t) < e.scrollTop;
					Xi(t, t.height + _i(r)),
					i && qn(o, null, r.height),
					o.curOp.forceUpdate = !0
				}
				return ! 0
			}),
			r
		}
		function Si(e, t, n, i) {
			e.text = t,
			e.stateAfter && (e.stateAfter = null),
			e.styles && (e.styles = null),
			null != e.order && (e.order = null),
			si(e),
			li(e, n);
			var r = i ? i(e) : 1;
			r != e.height && Xi(e, r)
		}
		function Mi(e) {
			e.parent = null,
			si(e)
		}
		function Ti(e, t) {
			if (e) for (;;) {
				var n = e.match(/(?:^|\s+)line-(background-)?(\S+)/);
				if (!n) break;
				e = e.slice(0, n.index) + e.slice(n.index + n[0].length);
				var i = n[1] ? "bgClass": "textClass";
				null == t[i] ? t[i] = n[2] : new RegExp("(?:^|s)" + n[2] + "(?:$|s)").test(t[i]) || (t[i] += " " + n[2])
			}
			return e
		}
		function Di(t, n) {
			if (t.blankLine) return t.blankLine(n);
			if (t.innerMode) {
				var i = e.innerMode(t, n);
				return i.mode.blankLine ? i.mode.blankLine(i.state) : void 0
			}
		}
		function Li(t, n, i, r) {
			for (var o = 0; 10 > o; o++) {
				r && (r[0] = e.innerMode(t, i).mode);
				var a = t.token(n, i);
				if (n.pos > n.start) return a
			}
			throw new Error("Mode " + t.name + " failed to advance stream.")
		}
		function Oi(e, t, n, i) {
			function r(e) {
				return {
					start: d.start,
					end: d.pos,
					string: d.current(),
					type: o || null,
					state: e ? ra(a.mode, u) : u
				}
			}
			var o, a = e.doc,
			s = a.mode;
			t = pe(a, t);
			var l, c = Vi(a, t.line),
			u = He(e, t.line, n),
			d = new da(c.text, e.options.tabSize);
			for (i && (l = []); (i || d.pos < t.ch) && !d.eol();) d.start = d.pos,
			o = Li(s, d, u),
			i && l.push(r(!0));
			return i ? l: r()
		}
		function Ni(e, t, n, i, r, o, a) {
			var s = n.flattenSpans;
			null == s && (s = e.options.flattenSpans);
			var l, c = 0,
			u = null,
			d = new da(t, e.options.tabSize),
			h = e.options.addModeClass && [null];
			for ("" == t && Ti(Di(n, i), o); ! d.eol();) {
				if (d.pos > e.options.maxHighlightLength ? (s = !1, a && $i(e, t, i, d.pos), d.pos = t.length, l = null) : l = Ti(Li(n, d, i, h), o), h) {
					var f = h[0].name;
					f && (l = "m-" + (l ? f + " " + l: f))
				}
				if (!s || u != l) {
					for (; c < d.start;) c = Math.min(d.start, c + 5e4),
					r(c, u);
					u = l
				}
				d.start = d.pos
			}
			for (; c < d.pos;) {
				var p = Math.min(d.pos, c + 5e4);
				r(p, u),
				c = p
			}
		}
		function Ai(e, t, n, i) {
			var r = [e.state.modeGen],
			o = {};
			Ni(e, t.text, e.doc.mode, n,
			function(e, t) {
				r.push(e, t)
			},
			o, i);
			for (var a = 0; a < e.state.overlays.length; ++a) {
				var s = e.state.overlays[a],
				l = 1,
				c = 0;
				Ni(e, t.text, s.mode, !0,
				function(e, t) {
					for (var n = l; e > c;) {
						var i = r[l];
						i > e && r.splice(l, 1, e, r[l + 1], i),
						l += 2,
						c = Math.min(e, i)
					}
					if (t) if (s.opaque) r.splice(n, l - n, e, "cm-overlay " + t),
					l = n + 2;
					else for (; l > n; n += 2) {
						var o = r[n + 1];
						r[n + 1] = (o ? o + " ": "") + "cm-overlay " + t
					}
				},
				o)
			}
			return {
				styles: r,
				classes: o.bgClass || o.textClass ? o: null
			}
		}
		function Ei(e, t, n) {
			if (!t.styles || t.styles[0] != e.state.modeGen) {
				var i = He(e, Qi(t)),
				r = Ai(e, t, t.text.length > e.options.maxHighlightLength ? ra(e.doc.mode, i) : i);
				t.stateAfter = i,
				t.styles = r.styles,
				r.classes ? t.styleClasses = r.classes: t.styleClasses && (t.styleClasses = null),
				n === e.doc.frontier && e.doc.frontier++
			}
			return t.styles
		}
		function $i(e, t, n, i) {
			var r = e.doc.mode,
			o = new da(t, e.options.tabSize);
			for (o.start = o.pos = i || 0, "" == t && Di(r, n); ! o.eol();) Li(r, o, n),
			o.start = o.pos
		}
		function qi(e, t) {
			if (!e || /^\s*$/.test(e)) return null;
			var n = t.addModeClass ? ya: va;
			return n[e] || (n[e] = e.replace(/\S+/g, "cm-$&"))
		}
		function ji(e, t) {
			var n = zr("span", null, null, yo ? "padding-right: .1px": null),
			i = {
				pre: zr("pre", [n], "CodeMirror-line"),
				content: n,
				col: 0,
				pos: 0,
				cm: e,
				splitSpaces: (go || yo) && e.getOption("lineWrapping")
			};
			t.measure = {};
			for (var r = 0; r <= (t.rest ? t.rest.length: 0); r++) {
				var o, a = r ? t.rest[r - 1] : t.line;
				i.pos = 0,
				i.addToken = Pi,
				Kr(e.display.measure) && (o = tr(a)) && (i.addToken = zi(i.addToken, o)),
				i.map = [],
				Hi(a, i, Ei(e, a, t != e.display.externalMeasured && Qi(a))),
				a.styleClasses && (a.styleClasses.bgClass && (i.bgClass = Rr(a.styleClasses.bgClass, i.bgClass || "")), a.styleClasses.textClass && (i.textClass = Rr(a.styleClasses.textClass, i.textClass || ""))),
				0 == i.map.length && i.map.push(0, 0, i.content.appendChild(Vr(e.display.measure))),
				0 == r ? (t.measure.map = i.map, t.measure.cache = {}) : ((t.measure.maps || (t.measure.maps = [])).push(i.map), (t.measure.caches || (t.measure.caches = [])).push({}))
			}
			return yo && /\bcm-tab\b/.test(i.content.lastChild.className) && (i.content.className = "cm-tab-wrap-hack"),
			La(e, "renderLine", e, t.line, i.pre),
			i.pre.className && (i.textClass = Rr(i.pre.className, i.textClass || "")),
			i
		}
		function Pi(e, t, n, i, r, o, a) {
			if (t) {
				var s = e.splitSpaces ? t.replace(/ {3,}/g, Ii) : t,
				l = e.cm.state.specialChars,
				c = !1;
				if (l.test(t)) for (var u = document.createDocumentFragment(), d = 0;;) {
					l.lastIndex = d;
					var h = l.exec(t),
					f = h ? h.index - d: t.length - d;
					if (f) {
						var p = document.createTextNode(s.slice(d, d + f));
						go && 9 > vo ? u.appendChild(zr("span", [p])) : u.appendChild(p),
						e.map.push(e.pos, e.pos + f, p),
						e.col += f,
						e.pos += f
					}
					if (!h) break;
					if (d += f + 1, "\t" == h[0]) {
						var m = e.cm.options.tabSize,
						g = m - e.col % m; (p = u.appendChild(zr("span", Dr(g), "cm-tab"))).setAttribute("role", "presentation"),
						p.setAttribute("cm-text", "\t"),
						e.col += g
					} else "\r" == h[0] || "\n" == h[0] ? ((p = u.appendChild(zr("span", "\r" == h[0] ? "␍": "␤", "cm-invalidchar"))).setAttribute("cm-text", h[0]), e.col += 1) : ((p = e.cm.options.specialCharPlaceholder(h[0])).setAttribute("cm-text", h[0]), go && 9 > vo ? u.appendChild(zr("span", [p])) : u.appendChild(p), e.col += 1);
					e.map.push(e.pos, e.pos + 1, p),
					e.pos++
				} else {
					e.col += t.length;
					u = document.createTextNode(s);
					e.map.push(e.pos, e.pos + t.length, u),
					go && 9 > vo && (c = !0),
					e.pos += t.length
				}
				if (n || i || r || c || a) {
					var v = n || "";
					i && (v += i),
					r && (v += r);
					var y = zr("span", [u], v, a);
					return o && (y.title = o),
					e.content.appendChild(y)
				}
				e.content.appendChild(u)
			}
		}
		function Ii(e) {
			for (var t = " ",
			n = 0; n < e.length - 2; ++n) t += n % 2 ? " ": " ";
			return t += " "
		}
		function zi(e, t) {
			return function(n, i, r, o, a, s, l) {
				r = r ? r + " cm-force-border": "cm-force-border";
				for (var c = n.pos,
				u = c + i.length;;) {
					for (var d = 0; d < t.length; d++) {
						var h = t[d];
						if (h.to > c && h.from <= c) break
					}
					if (h.to >= u) return e(n, i, r, o, a, s, l);
					e(n, i.slice(0, h.to - c), r, o, null, s, l),
					o = null,
					i = i.slice(h.to - c),
					c = h.to
				}
			}
		}
		function Wi(e, t, n, i) {
			var r = !i && n.widgetNode;
			r && e.map.push(e.pos, e.pos + t, r),
			!i && e.cm.display.input.needsContentAttribute && (r || (r = e.content.appendChild(document.createElement("span"))), r.setAttribute("cm-marker", n.id)),
			r && (e.cm.display.input.setUneditable(r), e.content.appendChild(r)),
			e.pos += t
		}
		function Hi(e, t, n) {
			var i = e.markedSpans,
			r = e.text,
			o = 0;
			if (i) for (var a, s, l, c, u, d, h, f = r.length,
			p = 0,
			m = 1,
			g = "",
			v = 0;;) {
				if (v == p) {
					l = c = u = d = s = "",
					h = null,
					v = 1 / 0;
					for (var y, b = [], w = 0; w < i.length; ++w) {
						var k = i[w],
						x = k.marker;
						"bookmark" == x.type && k.from == p && x.widgetNode ? b.push(x) : k.from <= p && (null == k.to || k.to > p || x.collapsed && k.to == p && k.from == p) ? (null != k.to && k.to != p && v > k.to && (v = k.to, c = ""), x.className && (l += " " + x.className), x.css && (s = (s ? s + ";": "") + x.css), x.startStyle && k.from == p && (u += " " + x.startStyle), x.endStyle && k.to == v && (y || (y = [])).push(x.endStyle, k.to), x.title && !d && (d = x.title), x.collapsed && (!h || di(h.marker, x) < 0) && (h = k)) : k.from > p && v > k.from && (v = k.from)
					}
					if (y) for (w = 0; w < y.length; w += 2) y[w + 1] == v && (c += " " + y[w]);
					if (!h || h.from == p) for (w = 0; w < b.length; ++w) Wi(t, 0, b[w]);
					if (h && (h.from || 0) == p) {
						if (Wi(t, (null == h.to ? f + 1 : h.to) - p, h.marker, null == h.from), null == h.to) return;
						h.to == p && (h = !1)
					}
				}
				if (p >= f) break;
				for (var _ = Math.min(f, v);;) {
					if (g) {
						var C = p + g.length;
						if (!h) {
							var S = C > _ ? g.slice(0, _ - p) : g;
							t.addToken(t, S, a ? a + l: l, u, p + S.length == v ? c: "", d, s)
						}
						if (C >= _) {
							g = g.slice(_ - p),
							p = _;
							break
						}
						p = C,
						u = ""
					}
					g = r.slice(o, o = n[m++]),
					a = qi(n[m++], t.cm.options)
				}
			} else for (m = 1; m < n.length; m += 2) t.addToken(t, r.slice(o, o = n[m]), qi(n[m + 1], t.cm.options))
		}
		function Fi(e, t) {
			return 0 == t.from.ch && 0 == t.to.ch && "" == Lr(t.text) && (!e.cm || e.cm.options.wholeLineUpdateBefore)
		}
		function Yi(e, t, n, i) {
			function r(e) {
				return n ? n[e] : null
			}
			function o(e, n, r) {
				Si(e, n, r, i),
				kr(e, "change", e, t)
			}
			function a(e, t) {
				for (var n = e,
				o = []; t > n; ++n) o.push(new ga(c[n], r(n), i));
				return o
			}
			var s = t.from,
			l = t.to,
			c = t.text,
			u = Vi(e, s.line),
			d = Vi(e, l.line),
			h = Lr(c),
			f = r(c.length - 1),
			p = l.line - s.line;
			if (t.full) e.insert(0, a(0, c.length)),
			e.remove(c.length, e.size - c.length);
			else if (Fi(e, t)) {
				m = a(0, c.length - 1);
				o(d, d.text, f),
				p && e.remove(s.line, p),
				m.length && e.insert(s.line, m)
			} else if (u == d) 1 == c.length ? o(u, u.text.slice(0, s.ch) + h + u.text.slice(l.ch), f) : ((m = a(1, c.length - 1)).push(new ga(h + u.text.slice(l.ch), f, i)), o(u, u.text.slice(0, s.ch) + c[0], r(0)), e.insert(s.line + 1, m));
			else if (1 == c.length) o(u, u.text.slice(0, s.ch) + c[0] + d.text.slice(l.ch), r(0)),
			e.remove(s.line + 1, p);
			else {
				o(u, u.text.slice(0, s.ch) + c[0], r(0)),
				o(d, h + d.text.slice(l.ch), f);
				var m = a(1, c.length - 1);
				p > 1 && e.remove(s.line + 1, p - 1),
				e.insert(s.line + 1, m)
			}
			kr(e, "change", e, t)
		}
		function Ri(e) {
			this.lines = e,
			this.parent = null;
			for (var t = 0,
			n = 0; t < e.length; ++t) e[t].parent = this,
			n += e[t].height;
			this.height = n
		}
		function Ui(e) {
			this.children = e;
			for (var t = 0,
			n = 0,
			i = 0; i < e.length; ++i) {
				var r = e[i];
				t += r.chunkSize(),
				n += r.height,
				r.parent = this
			}
			this.size = t,
			this.height = n,
			this.parent = null
		}
		function Bi(e, t, n) {
			function i(e, r, o) {
				if (e.linked) for (var a = 0; a < e.linked.length; ++a) {
					var s = e.linked[a];
					if (s.doc != r) {
						var l = o && s.sharedHist; (!n || l) && (t(s.doc, l), i(s.doc, e, l))
					}
				}
			}
			i(e, null, !0)
		}
		function Gi(e, t) {
			if (t.cm) throw new Error("This document is already in use.");
			e.doc = t,
			t.cm = e,
			o(e),
			n(e),
			e.options.lineWrapping || d(e),
			e.options.mode = t.modeOption,
			$t(e)
		}
		function Vi(e, t) {
			if (0 > (t -= e.first) || t >= e.size) throw new Error("There is no line " + (t + e.first) + " in the document.");
			for (var n = e; ! n.lines;) for (var i = 0;; ++i) {
				var r = n.children[i],
				o = r.chunkSize();
				if (o > t) {
					n = r;
					break
				}
				t -= o
			}
			return n.lines[t]
		}
		function Ki(e, t, n) {
			var i = [],
			r = t.line;
			return e.iter(t.line, n.line + 1,
			function(e) {
				var o = e.text;
				r == n.line && (o = o.slice(0, n.ch)),
				r == t.line && (o = o.slice(t.ch)),
				i.push(o),
				++r
			}),
			i
		}
		function Zi(e, t, n) {
			var i = [];
			return e.iter(t, n,
			function(e) {
				i.push(e.text)
			}),
			i
		}
		function Xi(e, t) {
			var n = t - e.height;
			if (n) for (var i = e; i; i = i.parent) i.height += n
		}
		function Qi(e) {
			if (null == e.parent) return null;
			for (var t = e.parent,
			n = Or(t.lines, e), i = t.parent; i; t = i, i = i.parent) for (var r = 0; i.children[r] != t; ++r) n += i.children[r].chunkSize();
			return n + t.first
		}
		function Ji(e, t) {
			var n = e.first;
			e: do {
				for (o = 0; o < e.children.length; ++o) {
					var i = e.children[o],
					r = i.height;
					if (r > t) {
						e = i;
						continue e
					}
					t -= r,
					n += i.chunkSize()
				}
				return n
			} while (! e . lines );
			for (var o = 0; o < e.lines.length; ++o) {
				var a = e.lines[o].height;
				if (a > t) break;
				t -= a
			}
			return n + o
		}
		function er(e) {
			for (var t = 0,
			n = (e = gi(e)).parent, i = 0; i < n.lines.length; ++i) {
				var r = n.lines[i];
				if (r == e) break;
				t += r.height
			}
			for (var o = n.parent; o; n = o, o = n.parent) for (i = 0; i < o.children.length; ++i) {
				var a = o.children[i];
				if (a == n) break;
				t += a.height
			}
			return t
		}
		function tr(e) {
			var t = e.order;
			return null == t && (t = e.order = is(e.text)),
			t
		}
		function nr(e) {
			this.done = [],
			this.undone = [],
			this.undoDepth = 1 / 0,
			this.lastModTime = this.lastSelTime = 0,
			this.lastOp = this.lastSelOp = null,
			this.lastOrigin = this.lastSelOrigin = null,
			this.generation = this.maxGeneration = e || 1
		}
		function ir(e, t) {
			var n = {
				from: B(t.from),
				to: Ko(t),
				text: Ki(e, t.from, t.to)
			};
			return ur(e, n, t.from.line, t.to.line + 1),
			Bi(e,
			function(e) {
				ur(e, n, t.from.line, t.to.line + 1)
			},
			!0),
			n
		}
		function rr(e) {
			for (; e.length && Lr(e).ranges;) e.pop()
		}
		function or(e, t) {
			return t ? (rr(e.done), Lr(e.done)) : e.done.length && !Lr(e.done).ranges ? Lr(e.done) : e.done.length > 1 && !e.done[e.done.length - 2].ranges ? (e.done.pop(), Lr(e.done)) : void 0
		}
		function ar(e, t, n, i) {
			var r = e.history;
			r.undone.length = 0;
			var o, a = +new Date;
			if ((r.lastOp == i || r.lastOrigin == t.origin && t.origin && ("+" == t.origin.charAt(0) && e.cm && r.lastModTime > a - e.cm.options.historyEventDelay || "*" == t.origin.charAt(0))) && (o = or(r, r.lastOp == i))) {
				var s = Lr(o.changes);
				0 == qo(t.from, t.to) && 0 == qo(t.from, s.to) ? s.to = Ko(t) : o.changes.push(ir(e, t))
			} else {
				var l = Lr(r.done);
				for (l && l.ranges || cr(e.sel, r.done), o = {
					changes: [ir(e, t)],
					generation: r.generation
				},
				r.done.push(o); r.done.length > r.undoDepth;) r.done.shift(),
				r.done[0].ranges || r.done.shift()
			}
			r.done.push(n),
			r.generation = ++r.maxGeneration,
			r.lastModTime = r.lastSelTime = a,
			r.lastOp = r.lastSelOp = i,
			r.lastOrigin = r.lastSelOrigin = t.origin,
			s || La(e, "historyAdded")
		}
		function sr(e, t, n, i) {
			var r = t.charAt(0);
			return "*" == r || "+" == r && n.ranges.length == i.ranges.length && n.somethingSelected() == i.somethingSelected() && new Date - e.history.lastSelTime <= (e.cm ? e.cm.options.historyEventDelay: 500)
		}
		function lr(e, t, n, i) {
			var r = e.history,
			o = i && i.origin;
			n == r.lastSelOp || o && r.lastSelOrigin == o && (r.lastModTime == r.lastSelTime && r.lastOrigin == o || sr(e, o, Lr(r.done), t)) ? r.done[r.done.length - 1] = t: cr(t, r.done),
			r.lastSelTime = +new Date,
			r.lastSelOrigin = o,
			r.lastSelOp = n,
			i && !1 !== i.clearRedo && rr(r.undone)
		}
		function cr(e, t) {
			var n = Lr(t);
			n && n.ranges && n.equals(e) || t.push(e)
		}
		function ur(e, t, n, i) {
			var r = t["spans_" + e.id],
			o = 0;
			e.iter(Math.max(e.first, n), Math.min(e.first + e.size, i),
			function(n) {
				n.markedSpans && ((r || (r = t["spans_" + e.id] = {}))[o] = n.markedSpans),
				++o
			})
		}
		function dr(e) {
			if (!e) return null;
			for (var t, n = 0; n < e.length; ++n) e[n].marker.explicitlyCleared ? t || (t = e.slice(0, n)) : t && t.push(e[n]);
			return t ? t.length ? t: null: e
		}
		function hr(e, t) {
			var n = t["spans_" + e.id];
			if (!n) return null;
			for (var i = 0,
			r = []; i < t.text.length; ++i) r.push(dr(n[i]));
			return r
		}
		function fr(e, t, n) {
			for (var i = 0,
			r = []; i < e.length; ++i) {
				var o = e[i];
				if (o.ranges) r.push(n ? ce.prototype.deepCopy.call(o) : o);
				else {
					var a = o.changes,
					s = [];
					r.push({
						changes: s
					});
					for (var l = 0; l < a.length; ++l) {
						var c, u = a[l];
						if (s.push({
							from: u.from,
							to: u.to,
							text: u.text
						}), t) for (var d in u)(c = d.match(/^spans_(\d+)$/)) && Or(t, Number(c[1])) > -1 && (Lr(s)[d] = u[d], delete u[d])
					}
				}
			}
			return r
		}
		function pr(e, t, n, i) {
			n < e.line ? e.line += i: t < e.line && (e.line = t, e.ch = 0)
		}
		function mr(e, t, n, i) {
			for (var r = 0; r < e.length; ++r) {
				var o = e[r],
				a = !0;
				if (o.ranges) {
					o.copied || (o = e[r] = o.deepCopy(), o.copied = !0);
					for (s = 0; s < o.ranges.length; s++) pr(o.ranges[s].anchor, t, n, i),
					pr(o.ranges[s].head, t, n, i)
				} else {
					for (var s = 0; s < o.changes.length; ++s) {
						var l = o.changes[s];
						if (n < l.from.line) l.from = $o(l.from.line + i, l.from.ch),
						l.to = $o(l.to.line + i, l.to.ch);
						else if (t <= l.to.line) {
							a = !1;
							break
						}
					}
					a || (e.splice(0, r + 1), r = 0)
				}
			}
		}
		function gr(e, t) {
			var n = t.from.line,
			i = t.to.line,
			r = t.text.length - (i - n) - 1;
			mr(e.done, n, i, r),
			mr(e.undone, n, i, r)
		}
		function vr(e) {
			return null != e.defaultPrevented ? e.defaultPrevented: 0 == e.returnValue
		}
		function yr(e) {
			return e.target || e.srcElement
		}
		function br(e) {
			var t = e.which;
			return null == t && (1 & e.button ? t = 1 : 2 & e.button ? t = 3 : 4 & e.button && (t = 2)),
			To && e.ctrlKey && 1 == t && (t = 3),
			t
		}
		function wr(e, t, n) {
			var i = e._handlers && e._handlers[t];
			return n ? i && i.length > 0 ? i.slice() : Ta: i || Ta
		}
		function kr(e, t) {
			var n = wr(e, t, !1);
			if (n.length) {
				var i, r = Array.prototype.slice.call(arguments, 2);
				Ho ? i = Ho.delayedCallbacks: Oa ? i = Oa: (i = Oa = [], setTimeout(xr, 0));
				for (var o = 0; o < n.length; ++o) i.push(function(e) {
					return function() {
						e.apply(null, r)
					}
				} (n[o]))
			}
		}
		function xr() {
			var e = Oa;
			Oa = null;
			for (var t = 0; t < e.length; ++t) e[t]()
		}
		function _r(e, t, n) {
			return "string" == typeof t && (t = {
				type: t,
				preventDefault: function() {
					this.defaultPrevented = !0
				}
			}),
			La(e, n || t.type, e, t),
			vr(t) || t.codemirrorIgnore
		}
		function Cr(e) {
			var t = e._handlers && e._handlers.cursorActivity;
			if (t) for (var n = e.curOp.cursorActivityHandlers || (e.curOp.cursorActivityHandlers = []), i = 0; i < t.length; ++i) - 1 == Or(n, t[i]) && n.push(t[i])
		}
		function Sr(e, t) {
			return wr(e, t).length > 0
		}
		function Mr(e) {
			e.prototype.on = function(e, t) {
				Ma(this, e, t)
			},
			e.prototype.off = function(e, t) {
				Da(this, e, t)
			}
		}
		function Tr() {
			this.id = null
		}
		function Dr(e) {
			for (; Ia.length <= e;) Ia.push(Lr(Ia) + " ");
			return Ia[e]
		}
		function Lr(e) {
			return e[e.length - 1]
		}
		function Or(e, t) {
			for (var n = 0; n < e.length; ++n) if (e[n] == t) return n;
			return - 1
		}
		function Nr(e, t) {
			for (var n = [], i = 0; i < e.length; i++) n[i] = t(e[i], i);
			return n
		}
		function Ar() {}
		function Er(e, t) {
			var n;
			return Object.create ? n = Object.create(e) : (Ar.prototype = e, n = new Ar),
			t && $r(t, n),
			n
		}
		function $r(e, t, n) {
			t || (t = {});
			for (var i in e) ! e.hasOwnProperty(i) || !1 === n && t.hasOwnProperty(i) || (t[i] = e[i]);
			return t
		}
		function qr(e) {
			var t = Array.prototype.slice.call(arguments, 1);
			return function() {
				return e.apply(null, t)
			}
		}
		function jr(e, t) {
			return t ? !!(t.source.indexOf("\\w") > -1 && Fa(e)) || t.test(e) : Fa(e)
		}
		function Pr(e) {
			for (var t in e) if (e.hasOwnProperty(t) && e[t]) return ! 1;
			return ! 0
		}
		function Ir(e) {
			return e.charCodeAt(0) >= 768 && Ya.test(e)
		}
		function zr(e, t, n, i) {
			var r = document.createElement(e);
			if (n && (r.className = n), i && (r.style.cssText = i), "string" == typeof t) r.appendChild(document.createTextNode(t));
			else if (t) for (var o = 0; o < t.length; ++o) r.appendChild(t[o]);
			return r
		}
		function Wr(e) {
			for (var t = e.childNodes.length; t > 0; --t) e.removeChild(e.firstChild);
			return e
		}
		function Hr(e, t) {
			return Wr(e).appendChild(t)
		}
		function Fr() {
			for (var e = document.activeElement; e && e.root && e.root.activeElement;) e = e.root.activeElement;
			return e
		}
		function Yr(e) {
			return new RegExp("(^|\\s)" + e + "(?:$|\\s)\\s*")
		}
		function Rr(e, t) {
			for (var n = e.split(" "), i = 0; i < n.length; i++) n[i] && !Yr(n[i]).test(t) && (t += " " + n[i]);
			return t
		}
		function Ur(e) {
			if (document.body.getElementsByClassName) for (var t = document.body.getElementsByClassName("CodeMirror"), n = 0; n < t.length; n++) {
				var i = t[n].CodeMirror;
				i && e(i)
			}
		}
		function Br() {
			Ka || (Gr(), Ka = !0)
		}
		function Gr() {
			var e;
			Ma(window, "resize",
			function() {
				null == e && (e = setTimeout(function() {
					e = null,
					Ur(Ft)
				},
				100))
			}),
			Ma(window, "blur",
			function() {
				Ur(gn)
			})
		}
		function Vr(e) {
			if (null == Ua) {
				var t = zr("span", "​");
				Hr(e, zr("span", [t, document.createTextNode("x")])),
				0 != e.firstChild.offsetHeight && (Ua = t.offsetWidth <= 1 && t.offsetHeight > 2 && !(go && 8 > vo))
			}
			var n = Ua ? zr("span", "​") : zr("span", " ", null, "display: inline-block; width: 1px; margin-right: -1px");
			return n.setAttribute("cm-text", ""),
			n
		}
		function Kr(e) {
			if (null != Ba) return Ba;
			var t = Hr(e, document.createTextNode("AخA")),
			n = Wa(t, 0, 1).getBoundingClientRect();
			if (!n || n.left == n.right) return ! 1;
			var i = Wa(t, 1, 2).getBoundingClientRect();
			return Ba = i.right - n.right < 3
		}
		function Zr(e) {
			if (null != es) return es;
			var t = Hr(e, zr("span", "x")),
			n = t.getBoundingClientRect(),
			i = Wa(t, 0, 1).getBoundingClientRect();
			return es = Math.abs(n.left - i.left) > 1
		}
		function Xr(e, t, n, i) {
			if (!e) return i(t, n, "ltr");
			for (var r = !1,
			o = 0; o < e.length; ++o) {
				var a = e[o]; (a.from < n && a.to > t || t == n && a.to == t) && (i(Math.max(a.from, t), Math.min(a.to, n), 1 == a.level ? "rtl": "ltr"), r = !0)
			}
			r || i(t, n, "ltr")
		}
		function Qr(e) {
			return e.level % 2 ? e.to: e.from
		}
		function Jr(e) {
			return e.level % 2 ? e.from: e.to
		}
		function eo(e) {
			var t = tr(e);
			return t ? Qr(t[0]) : 0
		}
		function to(e) {
			var t = tr(e);
			return t ? Jr(Lr(t)) : e.text.length
		}
		function no(e, t) {
			var n = Vi(e.doc, t),
			i = gi(n);
			i != n && (t = Qi(i));
			var r = tr(i),
			o = r ? r[0].level % 2 ? to(i) : eo(i) : 0;
			return $o(t, o)
		}
		function io(e, t) {
			for (var n, i = Vi(e.doc, t); n = pi(i);) i = n.find(1, !0).line,
			t = null;
			var r = tr(i),
			o = r ? r[0].level % 2 ? eo(i) : to(i) : i.text.length;
			return $o(null == t ? Qi(i) : t, o)
		}
		function ro(e, t) {
			var n = no(e, t.line),
			i = Vi(e.doc, n.line),
			r = tr(i);
			if (!r || 0 == r[0].level) {
				var o = Math.max(0, i.text.search(/\S/)),
				a = t.line == n.line && t.ch <= o && t.ch;
				return $o(n.line, a ? 0 : o)
			}
			return n
		}
		function oo(e, t, n) {
			var i = e[0].level;
			return t == i || n != i && n > t
		}
		function ao(e, t) {
			ns = null;
			for (var n, i = 0; i < e.length; ++i) {
				var r = e[i];
				if (r.from < t && r.to > t) return i;
				if (r.from == t || r.to == t) {
					if (null != n) return oo(e, r.level, e[n].level) ? (r.from != r.to && (ns = n), i) : (r.from != r.to && (ns = i), n);
					n = i
				}
			}
			return n
		}
		function so(e, t, n, i) {
			if (!i) return t + n;
			do {
				t += n
			} while ( t > 0 && Ir ( e . text . charAt ( t )));
			return t
		}
		function lo(e, t, n, i) {
			var r = tr(e);
			if (!r) return co(e, t, n, i);
			for (var o = ao(r, t), a = r[o], s = so(e, t, a.level % 2 ? -n: n, i);;) {
				if (s > a.from && s < a.to) return s;
				if (s == a.from || s == a.to) return ao(r, s) == o ? s: (a = r[o += n], n > 0 == a.level % 2 ? a.to: a.from);
				if (! (a = r[o += n])) return null;
				s = n > 0 == a.level % 2 ? so(e, a.to, -1, i) : so(e, a.from, 1, i)
			}
		}
		function co(e, t, n, i) {
			var r = t + n;
			if (i) for (; r > 0 && Ir(e.text.charAt(r));) r += n;
			return 0 > r || r > e.text.length ? null: r
		}
		var uo = navigator.userAgent,
		ho = navigator.platform,
		fo = /gecko\/\d/i.test(uo),
		po = /MSIE \d/.test(uo),
		mo = /Trident\/(?:[7-9]|\d{2,})\..*rv:(\d+)/.exec(uo),
		go = po || mo,
		vo = go && (po ? document.documentMode || 6 : mo[1]),
		yo = /WebKit\//.test(uo),
		bo = yo && /Qt\/\d+\.\d+/.test(uo),
		wo = /Chrome\//.test(uo),
		ko = /Opera\//.test(uo),
		xo = /Apple Computer/.test(navigator.vendor),
		_o = /Mac OS X 1\d\D([8-9]|\d\d)\D/.test(uo),
		Co = /PhantomJS/.test(uo),
		So = /AppleWebKit/.test(uo) && /Mobile\/\w+/.test(uo),
		Mo = So || /Android|webOS|BlackBerry|Opera Mini|Opera Mobi|IEMobile/i.test(uo),
		To = So || /Mac/.test(ho),
		Do = /win/i.test(ho),
		Lo = ko && uo.match(/Version\/(\d*\.\d*)/);
		Lo && (Lo = Number(Lo[1])),
		Lo && Lo >= 15 && (ko = !1, yo = !0);
		var Oo = To && (bo || ko && (null == Lo || 12.11 > Lo)),
		No = fo || go && vo >= 9,
		Ao = !1,
		Eo = !1;
		p.prototype = $r({
			update: function(e) {
				var t = e.scrollWidth > e.clientWidth + 1,
				n = e.scrollHeight > e.clientHeight + 1,
				i = e.nativeBarWidth;
				if (n) {
					this.vert.style.display = "block",
					this.vert.style.bottom = t ? i + "px": "0";
					var r = e.viewHeight - (t ? i: 0);
					this.vert.firstChild.style.height = Math.max(0, e.scrollHeight - e.clientHeight + r) + "px"
				} else this.vert.style.display = "",
				this.vert.firstChild.style.height = "0";
				if (t) {
					this.horiz.style.display = "block",
					this.horiz.style.right = n ? i + "px": "0",
					this.horiz.style.left = e.barLeft + "px";
					var o = e.viewWidth - e.barLeft - (n ? i: 0);
					this.horiz.firstChild.style.width = e.scrollWidth - e.clientWidth + o + "px"
				} else this.horiz.style.display = "",
				this.horiz.firstChild.style.width = "0";
				return ! this.checkedZeroWidth && e.clientHeight > 0 && (0 == i && this.zeroWidthHack(), this.checkedZeroWidth = !0),
				{
					right: n ? i: 0,
					bottom: t ? i: 0
				}
			},
			setScrollLeft: function(e) {
				this.horiz.scrollLeft != e && (this.horiz.scrollLeft = e),
				this.disableHoriz && this.enableZeroWidthBar(this.horiz, this.disableHoriz)
			},
			setScrollTop: function(e) {
				this.vert.scrollTop != e && (this.vert.scrollTop = e),
				this.disableVert && this.enableZeroWidthBar(this.vert, this.disableVert)
			},
			zeroWidthHack: function() {
				var e = To && !_o ? "12px": "18px";
				this.horiz.style.height = this.vert.style.width = e,
				this.horiz.style.pointerEvents = this.vert.style.pointerEvents = "none",
				this.disableHoriz = new Tr,
				this.disableVert = new Tr
			},
			enableZeroWidthBar: function(e, t) {
				function n() {
					var i = e.getBoundingClientRect();
					document.elementFromPoint(i.left + 1, i.bottom - 1) != e ? e.style.pointerEvents = "none": t.set(1e3, n)
				}
				e.style.pointerEvents = "auto",
				t.set(1e3, n)
			},
			clear: function() {
				var e = this.horiz.parentNode;
				e.removeChild(this.horiz),
				e.removeChild(this.vert)
			}
		},
		p.prototype),
		m.prototype = $r({
			update: function() {
				return {
					bottom: 0,
					right: 0
				}
			},
			setScrollLeft: function() {},
			setScrollTop: function() {},
			clear: function() {}
		},
		m.prototype),
		e.scrollbarModel = {
			native: p,
			null: m
		},
		C.prototype.signal = function(e, t) {
			Sr(e, t) && this.events.push(arguments)
		},
		C.prototype.finish = function() {
			for (var e = 0; e < this.events.length; e++) La.apply(null, this.events[e])
		};
		var $o = e.Pos = function(e, t) {
			return this instanceof $o ? (this.line = e, void(this.ch = t)) : new $o(e, t)
		},
		qo = e.cmpPos = function(e, t) {
			return e.line - t.line || e.ch - t.ch
		},
		jo = null;
		te.prototype = $r({
			init: function(e) {
				function t(e) {
					if (!_r(i, e)) {
						if (i.somethingSelected()) jo = i.getSelections(),
						n.inaccurateSelection && (n.prevInput = "", n.inaccurateSelection = !1, o.value = jo.join("\n"), za(o));
						else {
							if (!i.options.lineWiseCopyCut) return;
							var t = J(i);
							jo = t.text,
							"cut" == e.type ? i.setSelections(t.ranges, null, Ea) : (n.prevInput = "", o.value = t.text.join("\n"), za(o))
						}
						"cut" == e.type && (i.state.cutIncoming = !0)
					}
				}
				var n = this,
				i = this.cm,
				r = this.wrapper = ne(),
				o = this.textarea = r.firstChild;
				e.wrapper.insertBefore(r, e.wrapper.firstChild),
				So && (o.style.width = "0px"),
				Ma(o, "input",
				function() {
					go && vo >= 9 && n.hasSelection && (n.hasSelection = null),
					n.poll()
				}),
				Ma(o, "paste",
				function(e) {
					_r(i, e) || X(e, i) || (i.state.pasteIncoming = !0, n.fastPoll())
				}),
				Ma(o, "cut", t),
				Ma(o, "copy", t),
				Ma(e.scroller, "paste",
				function(t) {
					Yt(e, t) || _r(i, t) || (i.state.pasteIncoming = !0, n.focus())
				}),
				Ma(e.lineSpace, "selectstart",
				function(t) {
					Yt(e, t) || _a(t)
				}),
				Ma(o, "compositionstart",
				function() {
					var e = i.getCursor("from");
					n.composing && n.composing.range.clear(),
					n.composing = {
						start: e,
						range: i.markText(e, i.getCursor("to"), {
							className: "CodeMirror-composing"
						})
					}
				}),
				Ma(o, "compositionend",
				function() {
					n.composing && (n.poll(), n.composing.range.clear(), n.composing = null)
				})
			},
			prepareSelection: function() {
				var e = this.cm,
				t = e.display,
				n = e.doc,
				i = $e(e);
				if (e.options.moveInputWithCursor) {
					var r = ht(e, n.sel.primary().head, "div"),
					o = t.wrapper.getBoundingClientRect(),
					a = t.lineDiv.getBoundingClientRect();
					i.teTop = Math.max(0, Math.min(t.wrapper.clientHeight - 10, r.top + a.top - o.top)),
					i.teLeft = Math.max(0, Math.min(t.wrapper.clientWidth - 10, r.left + a.left - o.left))
				}
				return i
			},
			showSelection: function(e) {
				var t = this.cm.display;
				Hr(t.cursorDiv, e.cursors),
				Hr(t.selectionDiv, e.selection),
				null != e.teTop && (this.wrapper.style.top = e.teTop + "px", this.wrapper.style.left = e.teLeft + "px")
			},
			reset: function(e) {
				if (!this.contextMenuPending) {
					var t, n, i = this.cm,
					r = i.doc;
					if (i.somethingSelected()) {
						this.prevInput = "";
						var o = r.sel.primary(),
						a = (t = Ja && (o.to().line - o.from().line > 100 || (n = i.getSelection()).length > 1e3)) ? "-": n || i.getSelection();
						this.textarea.value = a,
						i.state.focused && za(this.textarea),
						go && vo >= 9 && (this.hasSelection = a)
					} else e || (this.prevInput = this.textarea.value = "", go && vo >= 9 && (this.hasSelection = null));
					this.inaccurateSelection = t
				}
			},
			getField: function() {
				return this.textarea
			},
			supportsTouch: function() {
				return ! 1
			},
			focus: function() {
				if ("nocursor" != this.cm.options.readOnly && (!Mo || Fr() != this.textarea)) try {
					this.textarea.focus()
				} catch(e) {}
			},
			blur: function() {
				this.textarea.blur()
			},
			resetPosition: function() {
				this.wrapper.style.top = this.wrapper.style.left = 0
			},
			receivedFocus: function() {
				this.slowPoll()
			},
			slowPoll: function() {
				var e = this;
				e.pollingFast || e.polling.set(this.cm.options.pollInterval,
				function() {
					e.poll(),
					e.cm.state.focused && e.slowPoll()
				})
			},
			fastPoll: function() {
				function e() {
					n.poll() || t ? (n.pollingFast = !1, n.slowPoll()) : (t = !0, n.polling.set(60, e))
				}
				var t = !1,
				n = this;
				n.pollingFast = !0,
				n.polling.set(20, e)
			},
			poll: function() {
				var e = this.cm,
				t = this.textarea,
				n = this.prevInput;
				if (this.contextMenuPending || !e.state.focused || Qa(t) && !n && !this.composing || e.isReadOnly() || e.options.disableInput || e.state.keySeq) return ! 1;
				var i = t.value;
				if (i == n && !e.somethingSelected()) return ! 1;
				if (go && vo >= 9 && this.hasSelection === i || To && /[\uf700-\uf7ff]/.test(i)) return e.display.input.reset(),
				!1;
				if (e.doc.sel == e.display.selForContextMenu) {
					var r = i.charCodeAt(0);
					if (8203 != r || n || (n = "​"), 8666 == r) return this.reset(),
					this.cm.execCommand("undo")
				}
				for (var o = 0,
				a = Math.min(n.length, i.length); a > o && n.charCodeAt(o) == i.charCodeAt(o);)++o;
				var s = this;
				return Dt(e,
				function() {
					Z(e, i.slice(o), n.length - o, null, s.composing ? "*compose": null),
					i.length > 1e3 || i.indexOf("\n") > -1 ? t.value = s.prevInput = "": s.prevInput = i,
					s.composing && (s.composing.range.clear(), s.composing.range = e.markText(s.composing.start, e.getCursor("to"), {
						className: "CodeMirror-composing"
					}))
				}),
				!0
			},
			ensurePolled: function() {
				this.pollingFast && this.poll() && (this.pollingFast = !1)
			},
			onKeyPress: function() {
				go && vo >= 9 && (this.hasSelection = null),
				this.fastPoll()
			},
			onContextMenu: function(e) {
				function t() {
					if (null != a.selectionStart) {
						var e = r.somethingSelected(),
						t = "​" + (e ? a.value: "");
						a.value = "⇚",
						a.value = t,
						i.prevInput = e ? "": "​",
						a.selectionStart = 1,
						a.selectionEnd = t.length,
						o.selForContextMenu = r.doc.sel
					}
				}
				function n() {
					if (i.contextMenuPending = !1, i.wrapper.style.cssText = u, a.style.cssText = c, go && 9 > vo && o.scrollbars.setScrollTop(o.scroller.scrollTop = l), null != a.selectionStart) { (!go || go && 9 > vo) && t();
						var e = 0,
						n = function() {
							o.selForContextMenu == r.doc.sel && 0 == a.selectionStart && a.selectionEnd > 0 && "​" == i.prevInput ? Lt(r, aa.selectAll)(r) : e++<10 ? o.detectingSelectAll = setTimeout(n, 500) : o.input.reset()
						};
						o.detectingSelectAll = setTimeout(n, 200)
					}
				}
				var i = this,
				r = i.cm,
				o = r.display,
				a = i.textarea,
				s = Rt(r, e),
				l = o.scroller.scrollTop;
				if (s && !ko) {
					r.options.resetSelectionOnContextMenu && -1 == r.doc.sel.contains(s) && Lt(r, Se)(r.doc, he(s), Ea);
					var c = a.style.cssText,
					u = i.wrapper.style.cssText;
					i.wrapper.style.cssText = "position: absolute";
					var d = i.wrapper.getBoundingClientRect();
					if (a.style.cssText = "position: absolute; width: 30px; height: 30px; top: " + (e.clientY - d.top - 5) + "px; left: " + (e.clientX - d.left - 5) + "px; z-index: 1000; background: " + (go ? "rgba(255, 255, 255, .05)": "transparent") + "; outline: none; border-width: 0; outline: none; overflow: hidden; opacity: .05; filter: alpha(opacity=5);", yo) var h = window.scrollY;
					if (o.input.focus(), yo && window.scrollTo(null, h), o.input.reset(), r.somethingSelected() || (a.value = i.prevInput = " "), i.contextMenuPending = !0, o.selForContextMenu = r.doc.sel, clearTimeout(o.detectingSelectAll), go && vo >= 9 && t(), No) {
						Sa(e);
						var f = function() {
							Da(window, "mouseup", f),
							setTimeout(n, 20)
						};
						Ma(window, "mouseup", f)
					} else setTimeout(n, 50)
				}
			},
			readOnlyChanged: function(e) {
				e || this.reset()
			},
			setUneditable: Ar,
			needsContentAttribute: !1
		},
		te.prototype),
		ie.prototype = $r({
			init: function(e) {
				function t(e) {
					if (!_r(i, e)) {
						if (i.somethingSelected()) jo = i.getSelections(),
						"cut" == e.type && i.replaceSelection("", null, "cut");
						else {
							if (!i.options.lineWiseCopyCut) return;
							var t = J(i);
							jo = t.text,
							"cut" == e.type && i.operation(function() {
								i.setSelections(t.ranges, 0, Ea),
								i.replaceSelection("", null, "cut")
							})
						}
						if (e.clipboardData && !So) e.preventDefault(),
						e.clipboardData.clearData(),
						e.clipboardData.setData("text/plain", jo.join("\n"));
						else {
							var n = ne(),
							r = n.firstChild;
							i.display.lineSpace.insertBefore(n, i.display.lineSpace.firstChild),
							r.value = jo.join("\n");
							var o = document.activeElement;
							za(r),
							setTimeout(function() {
								i.display.lineSpace.removeChild(n),
								o.focus()
							},
							50)
						}
					}
				}
				var n = this,
				i = n.cm,
				r = n.div = e.lineDiv;
				ee(r),
				Ma(r, "paste",
				function(e) {
					_r(i, e) || X(e, i)
				}),
				Ma(r, "compositionstart",
				function(e) {
					var t = e.data;
					if (n.composing = {
						sel: i.doc.sel,
						data: t,
						startData: t
					},
					t) {
						var r = i.doc.sel.primary(),
						o = i.getLine(r.head.line).indexOf(t, Math.max(0, r.head.ch - t.length));
						o > -1 && o <= r.head.ch && (n.composing.sel = he($o(r.head.line, o), $o(r.head.line, o + t.length)))
					}
				}),
				Ma(r, "compositionupdate",
				function(e) {
					n.composing.data = e.data
				}),
				Ma(r, "compositionend",
				function(e) {
					var t = n.composing;
					t && (e.data == t.startData || /\u200b/.test(e.data) || (t.data = e.data), setTimeout(function() {
						t.handled || n.applyComposition(t),
						n.composing == t && (n.composing = null)
					},
					50))
				}),
				Ma(r, "touchstart",
				function() {
					n.forceCompositionEnd()
				}),
				Ma(r, "input",
				function() {
					n.composing || (i.isReadOnly() || !n.pollContent()) && Dt(n.cm,
					function() {
						$t(i)
					})
				}),
				Ma(r, "copy", t),
				Ma(r, "cut", t)
			},
			prepareSelection: function() {
				var e = $e(this.cm, !1);
				return e.focus = this.cm.state.focused,
				e
			},
			showSelection: function(e) {
				e && this.cm.display.view.length && (e.focus && this.showPrimarySelection(), this.showMultipleSelections(e))
			},
			showPrimarySelection: function() {
				var e = window.getSelection(),
				t = this.cm.doc.sel.primary(),
				n = ae(this.cm, e.anchorNode, e.anchorOffset),
				i = ae(this.cm, e.focusNode, e.focusOffset);
				if (!n || n.bad || !i || i.bad || 0 != qo(V(n, i), t.from()) || 0 != qo(G(n, i), t.to())) {
					var r = re(this.cm, t.from()),
					o = re(this.cm, t.to());
					if (r || o) {
						var a = this.cm.display.view,
						s = e.rangeCount && e.getRangeAt(0);
						if (r) {
							if (!o) {
								var l = a[a.length - 1].measure,
								c = l.maps ? l.maps[l.maps.length - 1] : l.map;
								o = {
									node: c[c.length - 1],
									offset: c[c.length - 2] - c[c.length - 3]
								}
							}
						} else r = {
							node: a[0].measure.map[2],
							offset: 0
						};
						try {
							var u = Wa(r.node, r.offset, o.offset, o.node)
						} catch(e) {}
						u && (!fo && this.cm.state.focused ? (e.collapse(r.node, r.offset), u.collapsed || e.addRange(u)) : (e.removeAllRanges(), e.addRange(u)), s && null == e.anchorNode ? e.addRange(s) : fo && this.startGracePeriod()),
						this.rememberSelection()
					}
				}
			},
			startGracePeriod: function() {
				var e = this;
				clearTimeout(this.gracePeriod),
				this.gracePeriod = setTimeout(function() {
					e.gracePeriod = !1,
					e.selectionChanged() && e.cm.operation(function() {
						e.cm.curOp.selectionChanged = !0
					})
				},
				20)
			},
			showMultipleSelections: function(e) {
				Hr(this.cm.display.cursorDiv, e.cursors),
				Hr(this.cm.display.selectionDiv, e.selection)
			},
			rememberSelection: function() {
				var e = window.getSelection();
				this.lastAnchorNode = e.anchorNode,
				this.lastAnchorOffset = e.anchorOffset,
				this.lastFocusNode = e.focusNode,
				this.lastFocusOffset = e.focusOffset
			},
			selectionInEditor: function() {
				var e = window.getSelection();
				if (!e.rangeCount) return ! 1;
				var t = e.getRangeAt(0).commonAncestorContainer;
				return Ra(this.div, t)
			},
			focus: function() {
				"nocursor" != this.cm.options.readOnly && this.div.focus()
			},
			blur: function() {
				this.div.blur()
			},
			getField: function() {
				return this.div
			},
			supportsTouch: function() {
				return ! 0
			},
			receivedFocus: function() {
				function e() {
					t.cm.state.focused && (t.pollSelection(), t.polling.set(t.cm.options.pollInterval, e))
				}
				var t = this;
				this.selectionInEditor() ? this.pollSelection() : Dt(this.cm,
				function() {
					t.cm.curOp.selectionChanged = !0
				}),
				this.polling.set(this.cm.options.pollInterval, e)
			},
			selectionChanged: function() {
				var e = window.getSelection();
				return e.anchorNode != this.lastAnchorNode || e.anchorOffset != this.lastAnchorOffset || e.focusNode != this.lastFocusNode || e.focusOffset != this.lastFocusOffset
			},
			pollSelection: function() {
				if (!this.composing && !this.gracePeriod && this.selectionChanged()) {
					var e = window.getSelection(),
					t = this.cm;
					this.rememberSelection();
					var n = ae(t, e.anchorNode, e.anchorOffset),
					i = ae(t, e.focusNode, e.focusOffset);
					n && i && Dt(t,
					function() {
						Se(t.doc, he(n, i), Ea),
						(n.bad || i.bad) && (t.curOp.selectionChanged = !0)
					})
				}
			},
			pollContent: function() {
				var e = this.cm,
				t = e.display,
				n = e.doc.sel.primary(),
				i = n.from(),
				r = n.to();
				if (i.line < t.viewFrom || r.line > t.viewTo - 1) return ! 1;
				var o;
				if (i.line == t.viewFrom || 0 == (o = Pt(e, i.line))) var a = Qi(t.view[0].line),
				s = t.view[0].node;
				else var a = Qi(t.view[o].line),
				s = t.view[o - 1].node.nextSibling;
				var l = Pt(e, r.line);
				if (l == t.view.length - 1) var c = t.viewTo - 1,
				u = t.lineDiv.lastChild;
				else var c = Qi(t.view[l + 1].line) - 1,
				u = t.view[l + 1].node.previousSibling;
				for (var d = e.doc.splitLines(le(e, s, u, a, c)), h = Ki(e.doc, $o(a, 0), $o(c, Vi(e.doc, c).text.length)); d.length > 1 && h.length > 1;) if (Lr(d) == Lr(h)) d.pop(),
				h.pop(),
				c--;
				else {
					if (d[0] != h[0]) break;
					d.shift(),
					h.shift(),
					a++
				}
				for (var f = 0,
				p = 0,
				m = d[0], g = h[0], v = Math.min(m.length, g.length); v > f && m.charCodeAt(f) == g.charCodeAt(f);)++f;
				for (var y = Lr(d), b = Lr(h), w = Math.min(y.length - (1 == d.length ? f: 0), b.length - (1 == h.length ? f: 0)); w > p && y.charCodeAt(y.length - p - 1) == b.charCodeAt(b.length - p - 1);)++p;
				d[d.length - 1] = y.slice(0, y.length - p),
				d[0] = d[0].slice(f);
				var k = $o(a, f),
				x = $o(c, h.length ? Lr(h).length - p: 0);
				return d.length > 1 || d[0] || qo(k, x) ? (On(e.doc, d, k, x, "+input"), !0) : void 0
			},
			ensurePolled: function() {
				this.forceCompositionEnd()
			},
			reset: function() {
				this.forceCompositionEnd()
			},
			forceCompositionEnd: function() {
				this.composing && !this.composing.handled && (this.applyComposition(this.composing), this.composing.handled = !0, this.div.blur(), this.div.focus())
			},
			applyComposition: function(e) {
				this.cm.isReadOnly() ? Lt(this.cm, $t)(this.cm) : e.data && e.data != e.startData && Lt(this.cm, Z)(this.cm, e.data, 0, e.sel)
			},
			setUneditable: function(e) {
				e.contentEditable = "false"
			},
			onKeyPress: function(e) {
				e.preventDefault(),
				this.cm.isReadOnly() || Lt(this.cm, Z)(this.cm, String.fromCharCode(null == e.charCode ? e.keyCode: e.charCode), 0)
			},
			readOnlyChanged: function(e) {
				this.div.contentEditable = String("nocursor" != e)
			},
			onContextMenu: Ar,
			resetPosition: Ar,
			needsContentAttribute: !0
		},
		ie.prototype),
		e.inputStyles = {
			textarea: te,
			contenteditable: ie
		},
		ce.prototype = {
			primary: function() {
				return this.ranges[this.primIndex]
			},
			equals: function(e) {
				if (e == this) return ! 0;
				if (e.primIndex != this.primIndex || e.ranges.length != this.ranges.length) return ! 1;
				for (var t = 0; t < this.ranges.length; t++) {
					var n = this.ranges[t],
					i = e.ranges[t];
					if (0 != qo(n.anchor, i.anchor) || 0 != qo(n.head, i.head)) return ! 1
				}
				return ! 0
			},
			deepCopy: function() {
				for (var e = [], t = 0; t < this.ranges.length; t++) e[t] = new ue(B(this.ranges[t].anchor), B(this.ranges[t].head));
				return new ce(e, this.primIndex)
			},
			somethingSelected: function() {
				for (var e = 0; e < this.ranges.length; e++) if (!this.ranges[e].empty()) return ! 0;
				return ! 1
			},
			contains: function(e, t) {
				t || (t = e);
				for (var n = 0; n < this.ranges.length; n++) {
					var i = this.ranges[n];
					if (qo(t, i.from()) >= 0 && qo(e, i.to()) <= 0) return n
				}
				return - 1
			}
		},
		ue.prototype = {
			from: function() {
				return V(this.anchor, this.head)
			},
			to: function() {
				return G(this.anchor, this.head)
			},
			empty: function() {
				return this.head.line == this.anchor.line && this.head.ch == this.anchor.ch
			}
		};
		var Po, Io, zo, Wo = {
			left: 0,
			right: 0,
			top: 0,
			bottom: 0
		},
		Ho = null,
		Fo = 0,
		Yo = 0,
		Ro = 0,
		Uo = null;
		go ? Uo = -.53 : fo ? Uo = 15 : wo ? Uo = -.7 : xo && (Uo = -1 / 3);
		var Bo = function(e) {
			var t = e.wheelDeltaX,
			n = e.wheelDeltaY;
			return null == t && e.detail && e.axis == e.HORIZONTAL_AXIS && (t = e.detail),
			null == n && e.detail && e.axis == e.VERTICAL_AXIS ? n = e.detail: null == n && (n = e.wheelDelta),
			{
				x: t,
				y: n
			}
		};
		e.wheelEventPixels = function(e) {
			var t = Bo(e);
			return t.x *= Uo,
			t.y *= Uo,
			t
		};
		var Go = new Tr,
		Vo = null,
		Ko = e.changeEnd = function(e) {
			return e.text ? $o(e.from.line + e.text.length - 1, Lr(e.text).length + (1 == e.text.length ? e.from.ch: 0)) : e.to
		};
		e.prototype = {
			constructor: e,
			focus: function() {
				window.focus(),
				this.display.input.focus()
			},
			setOption: function(e, t) {
				var n = this.options,
				i = n[e]; (n[e] != t || "mode" == e) && (n[e] = t, Xo.hasOwnProperty(e) && Lt(this, Xo[e])(this, t, i))
			},
			getOption: function(e) {
				return this.options[e]
			},
			getDoc: function() {
				return this.doc
			},
			addKeyMap: function(e, t) {
				this.state.keyMaps[t ? "push": "unshift"](Un(e))
			},
			removeKeyMap: function(e) {
				for (var t = this.state.keyMaps,
				n = 0; n < t.length; ++n) if (t[n] == e || t[n].name == e) return t.splice(n, 1),
				!0
			},
			addOverlay: Ot(function(t, n) {
				var i = t.token ? t: e.getMode(this.options, t);
				if (i.startState) throw new Error("Overlays may not be stateful.");
				this.state.overlays.push({
					mode: i,
					modeSpec: t,
					opaque: n && n.opaque
				}),
				this.state.modeGen++,
				$t(this)
			}),
			removeOverlay: Ot(function(e) {
				for (var t = this.state.overlays,
				n = 0; n < t.length; ++n) {
					var i = t[n].modeSpec;
					if (i == e || "string" == typeof e && i.name == e) return t.splice(n, 1),
					this.state.modeGen++,
					void $t(this)
				}
			}),
			indentLine: Ot(function(e, t, n) {
				"string" != typeof t && "number" != typeof t && (t = null == t ? this.options.smartIndent ? "smart": "prev": t ? "add": "subtract"),
				ge(this.doc, e) && In(this, e, t, n)
			}),
			indentSelection: Ot(function(e) {
				for (var t = this.doc.sel.ranges,
				n = -1,
				i = 0; i < t.length; i++) {
					var r = t[i];
					if (r.empty()) r.head.line > n && (In(this, r.head.line, e, !0), n = r.head.line, i == this.doc.sel.primIndex && jn(this));
					else {
						var o = r.from(),
						a = r.to(),
						s = Math.max(n, o.line);
						n = Math.min(this.lastLine(), a.line - (a.ch ? 0 : 1)) + 1;
						for (var l = s; n > l; ++l) In(this, l, e);
						var c = this.doc.sel.ranges;
						0 == o.ch && t.length == c.length && c[i].from().ch > 0 && ke(this.doc, i, new ue(o, c[i].to()), Ea)
					}
				}
			}),
			getTokenAt: function(e, t) {
				return Oi(this, e, t)
			},
			getLineTokens: function(e, t) {
				return Oi(this, $o(e), t, !0)
			},
			getTokenTypeAt: function(e) {
				e = pe(this.doc, e);
				var t, n = Ei(this, Vi(this.doc, e.line)),
				i = 0,
				r = (n.length - 1) / 2,
				o = e.ch;
				if (0 == o) t = n[2];
				else for (;;) {
					var a = i + r >> 1;
					if ((a ? n[2 * a - 1] : 0) >= o) r = a;
					else {
						if (! (n[2 * a + 1] < o)) {
							t = n[2 * a + 2];
							break
						}
						i = a + 1
					}
				}
				var s = t ? t.indexOf("cm-overlay ") : -1;
				return 0 > s ? t: 0 == s ? null: t.slice(0, s - 1)
			},
			getModeAt: function(t) {
				var n = this.doc.mode;
				return n.innerMode ? e.innerMode(n, this.getTokenAt(t).state).mode: n
			},
			getHelper: function(e, t) {
				return this.getHelpers(e, t)[0]
			},
			getHelpers: function(e, t) {
				var n = [];
				if (!ia.hasOwnProperty(t)) return n;
				var i = ia[t],
				r = this.getModeAt(e);
				if ("string" == typeof r[t]) i[r[t]] && n.push(i[r[t]]);
				else if (r[t]) for (a = 0; a < r[t].length; a++) {
					var o = i[r[t][a]];
					o && n.push(o)
				} else r.helperType && i[r.helperType] ? n.push(i[r.helperType]) : i[r.name] && n.push(i[r.name]);
				for (var a = 0; a < i._global.length; a++) {
					var s = i._global[a];
					s.pred(r, this) && -1 == Or(n, s.val) && n.push(s.val)
				}
				return n
			},
			getStateAfter: function(e, t) {
				var n = this.doc;
				return e = fe(n, null == e ? n.first + n.size - 1 : e),
				He(this, e + 1, t)
			},
			cursorCoords: function(e, t) {
				var n, i = this.doc.sel.primary();
				return n = null == e ? i.head: "object" == typeof e ? pe(this.doc, e) : e ? i.from() : i.to(),
				ht(this, n, t || "page")
			},
			charCoords: function(e, t) {
				return dt(this, pe(this.doc, e), t || "page")
			},
			coordsChar: function(e, t) {
				return e = ut(this, e, t || "page"),
				mt(this, e.left, e.top)
			},
			lineAtHeight: function(e, t) {
				return e = ut(this, {
					top: e,
					left: 0
				},
				t || "page").top,
				Ji(this.doc, e + this.display.viewOffset)
			},
			heightAtLine: function(e, t) {
				var n, i = !1;
				if ("number" == typeof e) {
					var r = this.doc.first + this.doc.size - 1;
					e < this.doc.first ? e = this.doc.first: e > r && (e = r, i = !0),
					n = Vi(this.doc, e)
				} else n = e;
				return ct(this, n, {
					top: 0,
					left: 0
				},
				t || "page").top + (i ? this.doc.height - er(n) : 0)
			},
			defaultTextHeight: function() {
				return vt(this.display)
			},
			defaultCharWidth: function() {
				return yt(this.display)
			},
			setGutterMarker: Ot(function(e, t, n) {
				return zn(this.doc, e, "gutter",
				function(e) {
					var i = e.gutterMarkers || (e.gutterMarkers = {});
					return i[t] = n,
					!n && Pr(i) && (e.gutterMarkers = null),
					!0
				})
			}),
			clearGutter: Ot(function(e) {
				var t = this,
				n = t.doc,
				i = n.first;
				n.iter(function(n) {
					n.gutterMarkers && n.gutterMarkers[e] && (n.gutterMarkers[e] = null, qt(t, i, "gutter"), Pr(n.gutterMarkers) && (n.gutterMarkers = null)),
					++i
				})
			}),
			lineInfo: function(e) {
				if ("number" == typeof e) {
					if (!ge(this.doc, e)) return null;
					var t = e;
					if (! (e = Vi(this.doc, e))) return null
				} else if (null == (t = Qi(e))) return null;
				return {
					line: t,
					handle: e,
					text: e.text,
					gutterMarkers: e.gutterMarkers,
					textClass: e.textClass,
					bgClass: e.bgClass,
					wrapClass: e.wrapClass,
					widgets: e.widgets
				}
			},
			getViewport: function() {
				return {
					from: this.display.viewFrom,
					to: this.display.viewTo
				}
			},
			addWidget: function(e, t, n, i, r) {
				var o = this.display,
				a = (e = ht(this, pe(this.doc, e))).bottom,
				s = e.left;
				if (t.style.position = "absolute", t.setAttribute("cm-ignore-events", "true"), this.display.input.setUneditable(t), o.sizer.appendChild(t), "over" == i) a = e.top;
				else if ("above" == i || "near" == i) {
					var l = Math.max(o.wrapper.clientHeight, this.doc.height),
					c = Math.max(o.sizer.clientWidth, o.lineSpace.clientWidth); ("above" == i || e.bottom + t.offsetHeight > l) && e.top > t.offsetHeight ? a = e.top - t.offsetHeight: e.bottom + t.offsetHeight <= l && (a = e.bottom),
					s + t.offsetWidth > c && (s = c - t.offsetWidth)
				}
				t.style.top = a + "px",
				t.style.left = t.style.right = "",
				"right" == r ? (s = o.sizer.clientWidth - t.offsetWidth, t.style.right = "0px") : ("left" == r ? s = 0 : "middle" == r && (s = (o.sizer.clientWidth - t.offsetWidth) / 2), t.style.left = s + "px"),
				n && En(this, s, a, s + t.offsetWidth, a + t.offsetHeight)
			},
			triggerOnKeyDown: Ot(un),
			triggerOnKeyPress: Ot(fn),
			triggerOnKeyUp: hn,
			execCommand: function(e) {
				return aa.hasOwnProperty(e) ? aa[e].call(null, this) : void 0
			},
			triggerElectric: Ot(function(e) {
				Q(this, e)
			}),
			findPosH: function(e, t, n, i) {
				var r = 1;
				0 > t && (r = -1, t = -t);
				for (var o = 0,
				a = pe(this.doc, e); t > o && !(a = Hn(this.doc, a, r, n, i)).hitSide; ++o);
				return a
			},
			moveH: Ot(function(e, t) {
				var n = this;
				n.extendSelectionsBy(function(i) {
					return n.display.shift || n.doc.extend || i.empty() ? Hn(n.doc, i.head, e, t, n.options.rtlMoveVisually) : 0 > e ? i.from() : i.to()
				},
				qa)
			}),
			deleteH: Ot(function(e, t) {
				var n = this.doc.sel,
				i = this.doc;
				n.somethingSelected() ? i.replaceSelection("", null, "+delete") : Wn(this,
				function(n) {
					var r = Hn(i, n.head, e, t, !1);
					return 0 > e ? {
						from: r,
						to: n.head
					}: {
						from: n.head,
						to: r
					}
				})
			}),
			findPosV: function(e, t, n, i) {
				var r = 1,
				o = i;
				0 > t && (r = -1, t = -t);
				for (var a = 0,
				s = pe(this.doc, e); t > a; ++a) {
					var l = ht(this, s, "div");
					if (null == o ? o = l.left: l.left = o, (s = Fn(this, l, r, n)).hitSide) break
				}
				return s
			},
			moveV: Ot(function(e, t) {
				var n = this,
				i = this.doc,
				r = [],
				o = !n.display.shift && !i.extend && i.sel.somethingSelected();
				if (i.extendSelectionsBy(function(a) {
					if (o) return 0 > e ? a.from() : a.to();
					var s = ht(n, a.head, "div");
					null != a.goalColumn && (s.left = a.goalColumn),
					r.push(s.left);
					var l = Fn(n, s, e, t);
					return "page" == t && a == i.sel.primary() && qn(n, null, dt(n, l, "div").top - s.top),
					l
				},
				qa), r.length) for (var a = 0; a < i.sel.ranges.length; a++) i.sel.ranges[a].goalColumn = r[a]
			}),
			findWordAt: function(e) {
				var t = Vi(this.doc, e.line).text,
				n = e.ch,
				i = e.ch;
				if (t) {
					var r = this.getHelper(e, "wordChars"); (e.xRel < 0 || i == t.length) && n ? --n: ++i;
					for (var o = t.charAt(n), a = jr(o, r) ?
					function(e) {
						return jr(e, r)
					}: /\s/.test(o) ?
					function(e) {
						return /\s/.test(e)
					}: function(e) {
						return ! /\s/.test(e) && !jr(e)
					}; n > 0 && a(t.charAt(n - 1));)--n;
					for (; i < t.length && a(t.charAt(i));)++i
				}
				return new ue($o(e.line, n), $o(e.line, i))
			},
			toggleOverwrite: function(e) { (null == e || e != this.state.overwrite) && ((this.state.overwrite = !this.state.overwrite) ? Va(this.display.cursorDiv, "CodeMirror-overwrite") : Ga(this.display.cursorDiv, "CodeMirror-overwrite"), La(this, "overwriteToggle", this, this.state.overwrite))
			},
			hasFocus: function() {
				return this.display.input.getField() == Fr()
			},
			isReadOnly: function() {
				return ! (!this.options.readOnly && !this.doc.cantEdit)
			},
			scrollTo: Ot(function(e, t) { (null != e || null != t) && Pn(this),
				null != e && (this.curOp.scrollLeft = e),
				null != t && (this.curOp.scrollTop = t)
			}),
			getScrollInfo: function() {
				var e = this.display.scroller;
				return {
					left: e.scrollLeft,
					top: e.scrollTop,
					height: e.scrollHeight - Ue(this) - this.display.barHeight,
					width: e.scrollWidth - Ue(this) - this.display.barWidth,
					clientHeight: Ge(this),
					clientWidth: Be(this)
				}
			},
			scrollIntoView: Ot(function(e, t) {
				if (null == e ? (e = {
					from: this.doc.sel.primary().head,
					to: null
				},
				null == t && (t = this.options.cursorScrollMargin)) : "number" == typeof e ? e = {
					from: $o(e, 0),
					to: null
				}: null == e.from && (e = {
					from: e,
					to: null
				}), e.to || (e.to = e.from), e.margin = t || 0, null != e.from.line) Pn(this),
				this.curOp.scrollToPos = e;
				else {
					var n = $n(this, Math.min(e.from.left, e.to.left), Math.min(e.from.top, e.to.top) - e.margin, Math.max(e.from.right, e.to.right), Math.max(e.from.bottom, e.to.bottom) + e.margin);
					this.scrollTo(n.scrollLeft, n.scrollTop)
				}
			}),
			setSize: Ot(function(e, t) {
				function n(e) {
					return "number" == typeof e || /^\d+$/.test(String(e)) ? e + "px": e
				}
				var i = this;
				null != e && (i.display.wrapper.style.width = n(e)),
				null != t && (i.display.wrapper.style.height = n(t)),
				i.options.lineWrapping && ot(this);
				var r = i.display.viewFrom;
				i.doc.iter(r, i.display.viewTo,
				function(e) {
					if (e.widgets) for (var t = 0; t < e.widgets.length; t++) if (e.widgets[t].noHScroll) {
						qt(i, r, "widget");
						break
					}++r
				}),
				i.curOp.forceUpdate = !0,
				La(i, "refresh", this)
			}),
			operation: function(e) {
				return Dt(this, e)
			},
			refresh: Ot(function() {
				var e = this.display.cachedTextHeight;
				$t(this),
				this.curOp.forceUpdate = !0,
				at(this),
				this.scrollTo(this.doc.scrollLeft, this.doc.scrollTop),
				c(this),
				(null == e || Math.abs(e - vt(this.display)) > .5) && o(this),
				La(this, "refresh", this)
			}),
			swapDoc: Ot(function(e) {
				var t = this.doc;
				return t.cm = null,
				Gi(this, e),
				at(this),
				this.display.input.reset(),
				this.scrollTo(e.scrollLeft, e.scrollTop),
				this.curOp.forceScroll = !0,
				kr(this, "swapDoc", this, t),
				t
			}),
			getInputField: function() {
				return this.display.input.getField()
			},
			getWrapperElement: function() {
				return this.display.wrapper
			},
			getScrollerElement: function() {
				return this.display.scroller
			},
			getGutterElement: function() {
				return this.display.gutters
			}
		},
		Mr(e);
		var Zo = e.defaults = {},
		Xo = e.optionHandlers = {},
		Qo = e.Init = {
			toString: function() {
				return "CodeMirror.Init"
			}
		};
		Yn("value", "",
		function(e, t) {
			e.setValue(t)
		},
		!0),
		Yn("mode", null,
		function(e, t) {
			e.doc.modeOption = t,
			n(e)
		},
		!0),
		Yn("indentUnit", 2, n, !0),
		Yn("indentWithTabs", !1),
		Yn("smartIndent", !0),
		Yn("tabSize", 4,
		function(e) {
			i(e),
			at(e),
			$t(e)
		},
		!0),
		Yn("lineSeparator", null,
		function(e, t) {
			if (e.doc.lineSep = t, t) {
				var n = [],
				i = e.doc.first;
				e.doc.iter(function(e) {
					for (var r = 0;;) {
						var o = e.text.indexOf(t, r);
						if ( - 1 == o) break;
						r = o + t.length,
						n.push($o(i, o))
					}
					i++
				});
				for (var r = n.length - 1; r >= 0; r--) On(e.doc, t, n[r], $o(n[r].line, n[r].ch + t.length))
			}
		}),
		Yn("specialChars", /[\t\u0000-\u0019\u00ad\u200b-\u200f\u2028\u2029\ufeff]/g,
		function(t, n, i) {
			t.state.specialChars = new RegExp(n.source + (n.test("\t") ? "": "|\t"), "g"),
			i != e.Init && t.refresh()
		}),
		Yn("specialCharPlaceholder",
		function(e) {
			var t = zr("span", "•", "cm-invalidchar");
			return t.title = "\\u" + e.charCodeAt(0).toString(16),
			t.setAttribute("aria-label", t.title),
			t
		},
		function(e) {
			e.refresh()
		},
		!0),
		Yn("electricChars", !0),
		Yn("inputStyle", Mo ? "contenteditable": "textarea",
		function() {
			throw new Error("inputStyle can not (yet) be changed in a running editor")
		},
		!0),
		Yn("rtlMoveVisually", !Do),
		Yn("wholeLineUpdateBefore", !0),
		Yn("theme", "default",
		function(e) {
			a(e),
			s(e)
		},
		!0),
		Yn("keyMap", "default",
		function(t, n, i) {
			var r = Un(n),
			o = i != e.Init && Un(i);
			o && o.detach && o.detach(t, r),
			r.attach && r.attach(t, o || null)
		}),
		Yn("extraKeys", null),
		Yn("lineWrapping", !1,
		function(e) {
			e.options.lineWrapping ? (Va(e.display.wrapper, "CodeMirror-wrap"), e.display.sizer.style.minWidth = "", e.display.sizerWidth = null) : (Ga(e.display.wrapper, "CodeMirror-wrap"), d(e)),
			o(e),
			$t(e),
			at(e),
			setTimeout(function() {
				v(e)
			},
			100)
		},
		!0),
		Yn("gutters", [],
		function(e) {
			h(e.options),
			s(e)
		},
		!0),
		Yn("fixedGutter", !0,
		function(e, t) {
			e.display.gutters.style.left = t ? _(e.display) + "px": "0",
			e.refresh()
		},
		!0),
		Yn("coverGutterNextToScrollbar", !1,
		function(e) {
			v(e)
		},
		!0),
		Yn("scrollbarStyle", "native",
		function(e) {
			g(e),
			v(e),
			e.display.scrollbars.setScrollTop(e.doc.scrollTop),
			e.display.scrollbars.setScrollLeft(e.doc.scrollLeft)
		},
		!0),
		Yn("lineNumbers", !1,
		function(e) {
			h(e.options),
			s(e)
		},
		!0),
		Yn("firstLineNumber", 1, s, !0),
		Yn("lineNumberFormatter",
		function(e) {
			return e
		},
		s, !0),
		Yn("showCursorWhenSelecting", !1, Ee, !0),
		Yn("resetSelectionOnContextMenu", !0),
		Yn("lineWiseCopyCut", !0),
		Yn("readOnly", !1,
		function(e, t) {
			"nocursor" == t ? (gn(e), e.display.input.blur(), e.display.disabled = !0) : e.display.disabled = !1,
			e.display.input.readOnlyChanged(t)
		}),
		Yn("disableInput", !1,
		function(e, t) {
			t || e.display.input.reset()
		},
		!0),
		Yn("dragDrop", !0,
		function(t, n, i) {
			if (!n != !(i && i != e.Init)) {
				var r = t.display.dragFunctions,
				o = n ? Ma: Da;
				o(t.display.scroller, "dragstart", r.start),
				o(t.display.scroller, "dragenter", r.enter),
				o(t.display.scroller, "dragover", r.over),
				o(t.display.scroller, "dragleave", r.leave),
				o(t.display.scroller, "drop", r.drop)
			}
		}),
		Yn("allowDropFileTypes", null),
		Yn("cursorBlinkRate", 530),
		Yn("cursorScrollMargin", 0),
		Yn("cursorHeight", 1, Ee, !0),
		Yn("singleCursorHeightPerLine", !0, Ee, !0),
		Yn("workTime", 100),
		Yn("workDelay", 100),
		Yn("flattenSpans", !0, i, !0),
		Yn("addModeClass", !1, i, !0),
		Yn("pollInterval", 100),
		Yn("undoDepth", 200,
		function(e, t) {
			e.doc.history.undoDepth = t
		}),
		Yn("historyEventDelay", 1250),
		Yn("viewportMargin", 10,
		function(e) {
			e.refresh()
		},
		!0),
		Yn("maxHighlightLength", 1e4, i, !0),
		Yn("moveInputWithCursor", !0,
		function(e, t) {
			t || e.display.input.resetPosition()
		}),
		Yn("tabindex", null,
		function(e, t) {
			e.display.input.getField().tabIndex = t || ""
		}),
		Yn("autofocus", null);
		var Jo = e.modes = {},
		ea = e.mimeModes = {};
		e.defineMode = function(t, n) {
			e.defaults.mode || "null" == t || (e.defaults.mode = t),
			arguments.length > 2 && (n.dependencies = Array.prototype.slice.call(arguments, 2)),
			Jo[t] = n
		},
		e.defineMIME = function(e, t) {
			ea[e] = t
		},
		e.resolveMode = function(t) {
			if ("string" == typeof t && ea.hasOwnProperty(t)) t = ea[t];
			else if (t && "string" == typeof t.name && ea.hasOwnProperty(t.name)) {
				var n = ea[t.name];
				"string" == typeof n && (n = {
					name: n
				}),
				(t = Er(n, t)).name = n.name
			} else if ("string" == typeof t && /^[\w\-]+\/[\w\-]+\+xml$/.test(t)) return e.resolveMode("application/xml");
			return "string" == typeof t ? {
				name: t
			}: t || {
				name: "null"
			}
		},
		e.getMode = function(t, n) {
			var n = e.resolveMode(n),
			i = Jo[n.name];
			if (!i) return e.getMode(t, "text/plain");
			var r = i(t, n);
			if (ta.hasOwnProperty(n.name)) {
				var o = ta[n.name];
				for (var a in o) o.hasOwnProperty(a) && (r.hasOwnProperty(a) && (r["_" + a] = r[a]), r[a] = o[a])
			}
			if (r.name = n.name, n.helperType && (r.helperType = n.helperType), n.modeProps) for (var a in n.modeProps) r[a] = n.modeProps[a];
			return r
		},
		e.defineMode("null",
		function() {
			return {
				token: function(e) {
					e.skipToEnd()
				}
			}
		}),
		e.defineMIME("text/plain", "null");
		var ta = e.modeExtensions = {};
		e.extendMode = function(e, t) {
			$r(t, ta.hasOwnProperty(e) ? ta[e] : ta[e] = {})
		},
		e.defineExtension = function(t, n) {
			e.prototype[t] = n
		},
		e.defineDocExtension = function(e, t) {
			wa.prototype[e] = t
		},
		e.defineOption = Yn;
		var na = [];
		e.defineInitHook = function(e) {
			na.push(e)
		};
		var ia = e.helpers = {};
		e.registerHelper = function(t, n, i) {
			ia.hasOwnProperty(t) || (ia[t] = e[t] = {
				_global: []
			}),
			ia[t][n] = i
		},
		e.registerGlobalHelper = function(t, n, i, r) {
			e.registerHelper(t, n, r),
			ia[t]._global.push({
				pred: i,
				val: r
			})
		};
		var ra = e.copyState = function(e, t) {
			if (!0 === t) return t;
			if (e.copyState) return e.copyState(t);
			var n = {};
			for (var i in t) {
				var r = t[i];
				r instanceof Array && (r = r.concat([])),
				n[i] = r
			}
			return n
		},
		oa = e.startState = function(e, t, n) {
			return ! e.startState || e.startState(t, n)
		};
		e.innerMode = function(e, t) {
			for (; e.innerMode;) {
				var n = e.innerMode(t);
				if (!n || n.mode == e) break;
				t = n.state,
				e = n.mode
			}
			return n || {
				mode: e,
				state: t
			}
		};
		var aa = e.commands = {
			selectAll: function(e) {
				e.setSelection($o(e.firstLine(), 0), $o(e.lastLine()), Ea)
			},
			singleSelection: function(e) {
				e.setSelection(e.getCursor("anchor"), e.getCursor("head"), Ea)
			},
			killLine: function(e) {
				Wn(e,
				function(t) {
					if (t.empty()) {
						var n = Vi(e.doc, t.head.line).text.length;
						return t.head.ch == n && t.head.line < e.lastLine() ? {
							from: t.head,
							to: $o(t.head.line + 1, 0)
						}: {
							from: t.head,
							to: $o(t.head.line, n)
						}
					}
					return {
						from: t.from(),
						to: t.to()
					}
				})
			},
			deleteLine: function(e) {
				Wn(e,
				function(t) {
					return {
						from: $o(t.from().line, 0),
						to: pe(e.doc, $o(t.to().line + 1, 0))
					}
				})
			},
			delLineLeft: function(e) {
				Wn(e,
				function(e) {
					return {
						from: $o(e.from().line, 0),
						to: e.from()
					}
				})
			},
			delWrappedLineLeft: function(e) {
				Wn(e,
				function(t) {
					var n = e.charCoords(t.head, "div").top + 5;
					return {
						from: e.coordsChar({
							left: 0,
							top: n
						},
						"div"),
						to: t.from()
					}
				})
			},
			delWrappedLineRight: function(e) {
				Wn(e,
				function(t) {
					var n = e.charCoords(t.head, "div").top + 5,
					i = e.coordsChar({
						left: e.display.lineDiv.offsetWidth + 100,
						top: n
					},
					"div");
					return {
						from: t.from(),
						to: i
					}
				})
			},
			undo: function(e) {
				e.undo()
			},
			redo: function(e) {
				e.redo()
			},
			undoSelection: function(e) {
				e.undoSelection()
			},
			redoSelection: function(e) {
				e.redoSelection()
			},
			goDocStart: function(e) {
				e.extendSelection($o(e.firstLine(), 0))
			},
			goDocEnd: function(e) {
				e.extendSelection($o(e.lastLine()))
			},
			goLineStart: function(e) {
				e.extendSelectionsBy(function(t) {
					return no(e, t.head.line)
				},
				{
					origin: "+move",
					bias: 1
				})
			},
			goLineStartSmart: function(e) {
				e.extendSelectionsBy(function(t) {
					return ro(e, t.head)
				},
				{
					origin: "+move",
					bias: 1
				})
			},
			goLineEnd: function(e) {
				e.extendSelectionsBy(function(t) {
					return io(e, t.head.line)
				},
				{
					origin: "+move",
					bias: -1
				})
			},
			goLineRight: function(e) {
				e.extendSelectionsBy(function(t) {
					var n = e.charCoords(t.head, "div").top + 5;
					return e.coordsChar({
						left: e.display.lineDiv.offsetWidth + 100,
						top: n
					},
					"div")
				},
				qa)
			},
			goLineLeft: function(e) {
				e.extendSelectionsBy(function(t) {
					var n = e.charCoords(t.head, "div").top + 5;
					return e.coordsChar({
						left: 0,
						top: n
					},
					"div")
				},
				qa)
			},
			goLineLeftSmart: function(e) {
				e.extendSelectionsBy(function(t) {
					var n = e.charCoords(t.head, "div").top + 5,
					i = e.coordsChar({
						left: 0,
						top: n
					},
					"div");
					return i.ch < e.getLine(i.line).search(/\S/) ? ro(e, t.head) : i
				},
				qa)
			},
			goLineUp: function(e) {
				e.moveV( - 1, "line")
			},
			goLineDown: function(e) {
				e.moveV(1, "line")
			},
			goPageUp: function(e) {
				e.moveV( - 1, "page")
			},
			goPageDown: function(e) {
				e.moveV(1, "page")
			},
			goCharLeft: function(e) {
				e.moveH( - 1, "char")
			},
			goCharRight: function(e) {
				e.moveH(1, "char")
			},
			goColumnLeft: function(e) {
				e.moveH( - 1, "column")
			},
			goColumnRight: function(e) {
				e.moveH(1, "column")
			},
			goWordLeft: function(e) {
				e.moveH( - 1, "word")
			},
			goGroupRight: function(e) {
				e.moveH(1, "group")
			},
			goGroupLeft: function(e) {
				e.moveH( - 1, "group")
			},
			goWordRight: function(e) {
				e.moveH(1, "word")
			},
			delCharBefore: function(e) {
				e.deleteH( - 1, "char")
			},
			delCharAfter: function(e) {
				e.deleteH(1, "char")
			},
			delWordBefore: function(e) {
				e.deleteH( - 1, "word")
			},
			delWordAfter: function(e) {
				e.deleteH(1, "word")
			},
			delGroupBefore: function(e) {
				e.deleteH( - 1, "group")
			},
			delGroupAfter: function(e) {
				e.deleteH(1, "group")
			},
			indentAuto: function(e) {
				e.indentSelection("smart")
			},
			indentMore: function(e) {
				e.indentSelection("add")
			},
			indentLess: function(e) {
				e.indentSelection("subtract")
			},
			insertTab: function(e) {
				e.replaceSelection("\t")
			},
			insertSoftTab: function(e) {
				for (var t = [], n = e.listSelections(), i = e.options.tabSize, r = 0; r < n.length; r++) {
					var o = n[r].from(),
					a = ja(e.getLine(o.line), o.ch, i);
					t.push(new Array(i - a % i + 1).join(" "))
				}
				e.replaceSelections(t)
			},
			defaultTab: function(e) {
				e.somethingSelected() ? e.indentSelection("add") : e.execCommand("insertTab")
			},
			transposeChars: function(e) {
				Dt(e,
				function() {
					for (var t = e.listSelections(), n = [], i = 0; i < t.length; i++) {
						var r = t[i].head,
						o = Vi(e.doc, r.line).text;
						if (o) if (r.ch == o.length && (r = new $o(r.line, r.ch - 1)), r.ch > 0) r = new $o(r.line, r.ch + 1),
						e.replaceRange(o.charAt(r.ch - 1) + o.charAt(r.ch - 2), $o(r.line, r.ch - 2), r, "+transpose");
						else if (r.line > e.doc.first) {
							var a = Vi(e.doc, r.line - 1).text;
							a && e.replaceRange(o.charAt(0) + e.doc.lineSeparator() + a.charAt(a.length - 1), $o(r.line - 1, a.length - 1), $o(r.line, 1), "+transpose")
						}
						n.push(new ue(r, r))
					}
					e.setSelections(n)
				})
			},
			newlineAndIndent: function(e) {
				Dt(e,
				function() {
					for (var t = e.listSelections().length, n = 0; t > n; n++) {
						var i = e.listSelections()[n];
						e.replaceRange(e.doc.lineSeparator(), i.anchor, i.head, "+input"),
						e.indentLine(i.from().line + 1, null, !0)
					}
					jn(e)
				})
			},
			toggleOverwrite: function(e) {
				e.toggleOverwrite()
			}
		},
		sa = e.keyMap = {};
		sa.basic = {
			Left: "goCharLeft",
			Right: "goCharRight",
			Up: "goLineUp",
			Down: "goLineDown",
			End: "goLineEnd",
			Home: "goLineStartSmart",
			PageUp: "goPageUp",
			PageDown: "goPageDown",
			Delete: "delCharAfter",
			Backspace: "delCharBefore",
			"Shift-Backspace": "delCharBefore",
			Tab: "defaultTab",
			"Shift-Tab": "indentAuto",
			Enter: "newlineAndIndent",
			Insert: "toggleOverwrite",
			Esc: "singleSelection"
		},
		sa.pcDefault = {
			"Ctrl-A": "selectAll",
			"Ctrl-D": "deleteLine",
			"Ctrl-Z": "undo",
			"Shift-Ctrl-Z": "redo",
			"Ctrl-Y": "redo",
			"Ctrl-Home": "goDocStart",
			"Ctrl-End": "goDocEnd",
			"Ctrl-Up": "goLineUp",
			"Ctrl-Down": "goLineDown",
			"Ctrl-Left": "goGroupLeft",
			"Ctrl-Right": "goGroupRight",
			"Alt-Left": "goLineStart",
			"Alt-Right": "goLineEnd",
			"Ctrl-Backspace": "delGroupBefore",
			"Ctrl-Delete": "delGroupAfter",
			"Ctrl-S": "save",
			"Ctrl-F": "find",
			"Ctrl-G": "findNext",
			"Shift-Ctrl-G": "findPrev",
			"Shift-Ctrl-F": "replace",
			"Shift-Ctrl-R": "replaceAll",
			"Ctrl-[": "indentLess",
			"Ctrl-]": "indentMore",
			"Ctrl-U": "undoSelection",
			"Shift-Ctrl-U": "redoSelection",
			"Alt-U": "redoSelection",
			fallthrough: "basic"
		},
		sa.emacsy = {
			"Ctrl-F": "goCharRight",
			"Ctrl-B": "goCharLeft",
			"Ctrl-P": "goLineUp",
			"Ctrl-N": "goLineDown",
			"Alt-F": "goWordRight",
			"Alt-B": "goWordLeft",
			"Ctrl-A": "goLineStart",
			"Ctrl-E": "goLineEnd",
			"Ctrl-V": "goPageDown",
			"Shift-Ctrl-V": "goPageUp",
			"Ctrl-D": "delCharAfter",
			"Ctrl-H": "delCharBefore",
			"Alt-D": "delWordAfter",
			"Alt-Backspace": "delWordBefore",
			"Ctrl-K": "killLine",
			"Ctrl-T": "transposeChars"
		},
		sa.macDefault = {
			"Cmd-A": "selectAll",
			"Cmd-D": "deleteLine",
			"Cmd-Z": "undo",
			"Shift-Cmd-Z": "redo",
			"Cmd-Y": "redo",
			"Cmd-Home": "goDocStart",
			"Cmd-Up": "goDocStart",
			"Cmd-End": "goDocEnd",
			"Cmd-Down": "goDocEnd",
			"Alt-Left": "goGroupLeft",
			"Alt-Right": "goGroupRight",
			"Cmd-Left": "goLineLeft",
			"Cmd-Right": "goLineRight",
			"Alt-Backspace": "delGroupBefore",
			"Ctrl-Alt-Backspace": "delGroupAfter",
			"Alt-Delete": "delGroupAfter",
			"Cmd-S": "save",
			"Cmd-F": "find",
			"Cmd-G": "findNext",
			"Shift-Cmd-G": "findPrev",
			"Cmd-Alt-F": "replace",
			"Shift-Cmd-Alt-F": "replaceAll",
			"Cmd-[": "indentLess",
			"Cmd-]": "indentMore",
			"Cmd-Backspace": "delWrappedLineLeft",
			"Cmd-Delete": "delWrappedLineRight",
			"Cmd-U": "undoSelection",
			"Shift-Cmd-U": "redoSelection",
			"Ctrl-Up": "goDocStart",
			"Ctrl-Down": "goDocEnd",
			fallthrough: ["basic", "emacsy"]
		},
		sa.
	default = To ? sa.macDefault: sa.pcDefault,
		e.normalizeKeyMap = function(e) {
			var t = {};
			for (var n in e) if (e.hasOwnProperty(n)) {
				var i = e[n];
				if (/^(name|fallthrough|(de|at)tach)$/.test(n)) continue;
				if ("..." == i) {
					delete e[n];
					continue
				}
				for (var r = Nr(n.split(" "), Rn), o = 0; o < r.length; o++) {
					var a, s;
					o == r.length - 1 ? (s = r.join(" "), a = i) : (s = r.slice(0, o + 1).join(" "), a = "...");
					var l = t[s];
					if (l) {
						if (l != a) throw new Error("Inconsistent bindings for " + s)
					} else t[s] = a
				}
				delete e[n]
			}
			for (var c in t) e[c] = t[c];
			return e
		};
		var la = e.lookupKey = function(e, t, n, i) {
			var r = (t = Un(t)).call ? t.call(e, i) : t[e];
			if (!1 === r) return "nothing";
			if ("..." === r) return "multi";
			if (null != r && n(r)) return "handled";
			if (t.fallthrough) {
				if ("[object Array]" != Object.prototype.toString.call(t.fallthrough)) return la(e, t.fallthrough, n, i);
				for (var o = 0; o < t.fallthrough.length; o++) {
					var a = la(e, t.fallthrough[o], n, i);
					if (a) return a
				}
			}
		},
		ca = e.isModifierKey = function(e) {
			var t = "string" == typeof e ? e: ts[e.keyCode];
			return "Ctrl" == t || "Alt" == t || "Shift" == t || "Mod" == t
		},
		ua = e.keyName = function(e, t) {
			if (ko && 34 == e.keyCode && e.char) return ! 1;
			var n = ts[e.keyCode],
			i = n;
			return null != i && !e.altGraphKey && (e.altKey && "Alt" != n && (i = "Alt-" + i), (Oo ? e.metaKey: e.ctrlKey) && "Ctrl" != n && (i = "Ctrl-" + i), (Oo ? e.ctrlKey: e.metaKey) && "Cmd" != n && (i = "Cmd-" + i), !t && e.shiftKey && "Shift" != n && (i = "Shift-" + i), i)
		};
		e.fromTextArea = function(t, n) {
			function i() {
				t.value = l.getValue()
			}
			if (n = n ? $r(n) : {},
			n.value = t.value, !n.tabindex && t.tabIndex && (n.tabindex = t.tabIndex), !n.placeholder && t.placeholder && (n.placeholder = t.placeholder), null == n.autofocus) {
				var r = Fr();
				n.autofocus = r == t || null != t.getAttribute("autofocus") && r == document.body
			}
			if (t.form && (Ma(t.form, "submit", i), !n.leaveSubmitMethodAlone)) {
				var o = t.form,
				a = o.submit;
				try {
					var s = o.submit = function() {
						i(),
						o.submit = a,
						o.submit(),
						o.submit = s
					}
				} catch(e) {}
			}
			n.finishInit = function(e) {
				e.save = i,
				e.getTextArea = function() {
					return t
				},
				e.toTextArea = function() {
					e.toTextArea = isNaN,
					i(),
					t.parentNode.removeChild(e.getWrapperElement()),
					t.style.display = "",
					t.form && (Da(t.form, "submit", i), "function" == typeof t.form.submit && (t.form.submit = a))
				}
			},
			t.style.display = "none";
			var l = e(function(e) {
				t.parentNode.insertBefore(e, t.nextSibling)
			},
			n);
			return l
		};
		var da = e.StringStream = function(e, t) {
			this.pos = this.start = 0,
			this.string = e,
			this.tabSize = t || 8,
			this.lastColumnPos = this.lastColumnValue = 0,
			this.lineStart = 0
		};
		da.prototype = {
			eol: function() {
				return this.pos >= this.string.length
			},
			sol: function() {
				return this.pos == this.lineStart
			},
			peek: function() {
				return this.string.charAt(this.pos) || void 0
			},
			next: function() {
				return this.pos < this.string.length ? this.string.charAt(this.pos++) : void 0
			},
			eat: function(e) {
				var t = this.string.charAt(this.pos);
				if ("string" == typeof e) n = t == e;
				else var n = t && (e.test ? e.test(t) : e(t));
				return n ? (++this.pos, t) : void 0
			},
			eatWhile: function(e) {
				for (var t = this.pos; this.eat(e););
				return this.pos > t
			},
			eatSpace: function() {
				for (var e = this.pos;
				/[\s\u00a0]/.test(this.string.charAt(this.pos));)++this.pos;
				return this.pos > e
			},
			skipToEnd: function() {
				this.pos = this.string.length
			},
			skipTo: function(e) {
				var t = this.string.indexOf(e, this.pos);
				return t > -1 ? (this.pos = t, !0) : void 0
			},
			backUp: function(e) {
				this.pos -= e
			},
			column: function() {
				return this.lastColumnPos < this.start && (this.lastColumnValue = ja(this.string, this.start, this.tabSize, this.lastColumnPos, this.lastColumnValue), this.lastColumnPos = this.start),
				this.lastColumnValue - (this.lineStart ? ja(this.string, this.lineStart, this.tabSize) : 0)
			},
			indentation: function() {
				return ja(this.string, null, this.tabSize) - (this.lineStart ? ja(this.string, this.lineStart, this.tabSize) : 0)
			},
			match: function(e, t, n) {
				if ("string" != typeof e) {
					var i = this.string.slice(this.pos).match(e);
					return i && i.index > 0 ? null: (i && !1 !== t && (this.pos += i[0].length), i)
				}
				var r = function(e) {
					return n ? e.toLowerCase() : e
				};
				return r(this.string.substr(this.pos, e.length)) == r(e) ? (!1 !== t && (this.pos += e.length), !0) : void 0
			},
			current: function() {
				return this.string.slice(this.start, this.pos)
			},
			hideFirstChars: function(e, t) {
				this.lineStart += e;
				try {
					return t()
				} finally {
					this.lineStart -= e
				}
			}
		};
		var ha = 0,
		fa = e.TextMarker = function(e, t) {
			this.lines = [],
			this.type = t,
			this.doc = e,
			this.id = ++ha
		};
		Mr(fa),
		fa.prototype.clear = function() {
			if (!this.explicitlyCleared) {
				var e = this.doc.cm,
				t = e && !e.curOp;
				if (t && bt(e), Sr(this, "clear")) {
					var n = this.find();
					n && kr(this, "clear", n.from, n.to)
				}
				for (var i = null,
				r = null,
				o = 0; o < this.lines.length; ++o) {
					var a = this.lines[o],
					s = Qn(a.markedSpans, this);
					e && !this.collapsed ? qt(e, Qi(a), "text") : e && (null != s.to && (r = Qi(a)), null != s.from && (i = Qi(a))),
					a.markedSpans = Jn(a.markedSpans, s),
					null == s.from && this.collapsed && !wi(this.doc, a) && e && Xi(a, vt(e.display))
				}
				if (e && this.collapsed && !e.options.lineWrapping) for (o = 0; o < this.lines.length; ++o) {
					var l = gi(this.lines[o]),
					c = u(l);
					c > e.display.maxLineLength && (e.display.maxLine = l, e.display.maxLineLength = c, e.display.maxLineChanged = !0)
				}
				null != i && e && this.collapsed && $t(e, i, r + 1),
				this.lines.length = 0,
				this.explicitlyCleared = !0,
				this.atomic && this.doc.cantEdit && (this.doc.cantEdit = !1, e && De(e.doc)),
				e && kr(e, "markerCleared", e, this),
				t && kt(e),
				this.parent && this.parent.clear()
			}
		},
		fa.prototype.find = function(e, t) {
			null == e && "bookmark" == this.type && (e = 1);
			for (var n, i, r = 0; r < this.lines.length; ++r) {
				var o = this.lines[r],
				a = Qn(o.markedSpans, this);
				if (null != a.from && (n = $o(t ? o: Qi(o), a.from), -1 == e)) return n;
				if (null != a.to && (i = $o(t ? o: Qi(o), a.to), 1 == e)) return i
			}
			return n && {
				from: n,
				to: i
			}
		},
		fa.prototype.changed = function() {
			var e = this.find( - 1, !0),
			t = this,
			n = this.doc.cm;
			e && n && Dt(n,
			function() {
				var i = e.line,
				r = Qi(e.line),
				o = Qe(n, r);
				if (o && (rt(o), n.curOp.selectionChanged = n.curOp.forceUpdate = !0), n.curOp.updateMaxLine = !0, !wi(t.doc, i) && null != t.height) {
					var a = t.height;
					t.height = null;
					var s = _i(t) - a;
					s && Xi(i, i.height + s)
				}
			})
		},
		fa.prototype.attachLine = function(e) {
			if (!this.lines.length && this.doc.cm) {
				var t = this.doc.cm.curOp;
				t.maybeHiddenMarkers && -1 != Or(t.maybeHiddenMarkers, this) || (t.maybeUnhiddenMarkers || (t.maybeUnhiddenMarkers = [])).push(this)
			}
			this.lines.push(e)
		},
		fa.prototype.detachLine = function(e) {
			if (this.lines.splice(Or(this.lines, e), 1), !this.lines.length && this.doc.cm) {
				var t = this.doc.cm.curOp; (t.maybeHiddenMarkers || (t.maybeHiddenMarkers = [])).push(this)
			}
		};
		var ha = 0,
		pa = e.SharedTextMarker = function(e, t) {
			this.markers = e,
			this.primary = t;
			for (var n = 0; n < e.length; ++n) e[n].parent = this
		};
		Mr(pa),
		pa.prototype.clear = function() {
			if (!this.explicitlyCleared) {
				this.explicitlyCleared = !0;
				for (var e = 0; e < this.markers.length; ++e) this.markers[e].clear();
				kr(this, "clear")
			}
		},
		pa.prototype.find = function(e, t) {
			return this.primary.find(e, t)
		};
		var ma = e.LineWidget = function(e, t, n) {
			if (n) for (var i in n) n.hasOwnProperty(i) && (this[i] = n[i]);
			this.doc = e,
			this.node = t
		};
		Mr(ma),
		ma.prototype.clear = function() {
			var e = this.doc.cm,
			t = this.line.widgets,
			n = this.line,
			i = Qi(n);
			if (null != i && t) {
				for (var r = 0; r < t.length; ++r) t[r] == this && t.splice(r--, 1);
				t.length || (n.widgets = null);
				var o = _i(this);
				Xi(n, Math.max(0, n.height - o)),
				e && Dt(e,
				function() {
					xi(e, n, -o),
					qt(e, i, "widget")
				})
			}
		},
		ma.prototype.changed = function() {
			var e = this.height,
			t = this.doc.cm,
			n = this.line;
			this.height = null;
			var i = _i(this) - e;
			i && (Xi(n, n.height + i), t && Dt(t,
			function() {
				t.curOp.forceUpdate = !0,
				xi(t, n, i)
			}))
		};
		var ga = e.Line = function(e, t, n) {
			this.text = e,
			li(this, t),
			this.height = n ? n(this) : 1
		};
		Mr(ga),
		ga.prototype.lineNo = function() {
			return Qi(this)
		};
		var va = {},
		ya = {};
		Ri.prototype = {
			chunkSize: function() {
				return this.lines.length
			},
			removeInner: function(e, t) {
				for (var n = e,
				i = e + t; i > n; ++n) {
					var r = this.lines[n];
					this.height -= r.height,
					Mi(r),
					kr(r, "delete")
				}
				this.lines.splice(e, t)
			},
			collapse: function(e) {
				e.push.apply(e, this.lines)
			},
			insertInner: function(e, t, n) {
				this.height += n,
				this.lines = this.lines.slice(0, e).concat(t).concat(this.lines.slice(e));
				for (var i = 0; i < t.length; ++i) t[i].parent = this
			},
			iterN: function(e, t, n) {
				for (var i = e + t; i > e; ++e) if (n(this.lines[e])) return ! 0
			}
		},
		Ui.prototype = {
			chunkSize: function() {
				return this.size
			},
			removeInner: function(e, t) {
				this.size -= t;
				for (var n = 0; n < this.children.length; ++n) {
					var i = this.children[n],
					r = i.chunkSize();
					if (r > e) {
						var o = Math.min(t, r - e),
						a = i.height;
						if (i.removeInner(e, o), this.height -= a - i.height, r == o && (this.children.splice(n--, 1), i.parent = null), 0 == (t -= o)) break;
						e = 0
					} else e -= r
				}
				if (this.size - t < 25 && (this.children.length > 1 || !(this.children[0] instanceof Ri))) {
					var s = [];
					this.collapse(s),
					this.children = [new Ri(s)],
					this.children[0].parent = this
				}
			},
			collapse: function(e) {
				for (var t = 0; t < this.children.length; ++t) this.children[t].collapse(e)
			},
			insertInner: function(e, t, n) {
				this.size += t.length,
				this.height += n;
				for (var i = 0; i < this.children.length; ++i) {
					var r = this.children[i],
					o = r.chunkSize();
					if (o >= e) {
						if (r.insertInner(e, t, n), r.lines && r.lines.length > 50) {
							for (; r.lines.length > 50;) {
								var a = new Ri(r.lines.splice(r.lines.length - 25, 25));
								r.height -= a.height,
								this.children.splice(i + 1, 0, a),
								a.parent = this
							}
							this.maybeSpill()
						}
						break
					}
					e -= o
				}
			},
			maybeSpill: function() {
				if (! (this.children.length <= 10)) {
					var e = this;
					do {
						var t = new Ui(e.children.splice(e.children.length - 5, 5));
						if (e.parent) {
							e.size -= t.size,
							e.height -= t.height;
							var n = Or(e.parent.children, e);
							e.parent.children.splice(n + 1, 0, t)
						} else {
							var i = new Ui(e.children);
							i.parent = e,
							e.children = [i, t],
							e = i
						}
						t.parent = e.parent
					} while ( e . children . length > 10 );
					e.parent.maybeSpill()
				}
			},
			iterN: function(e, t, n) {
				for (var i = 0; i < this.children.length; ++i) {
					var r = this.children[i],
					o = r.chunkSize();
					if (o > e) {
						var a = Math.min(t, o - e);
						if (r.iterN(e, a, n)) return ! 0;
						if (0 == (t -= a)) break;
						e = 0
					} else e -= o
				}
			}
		};
		var ba = 0,
		wa = e.Doc = function(e, t, n, i) {
			if (! (this instanceof wa)) return new wa(e, t, n, i);
			null == n && (n = 0),
			Ui.call(this, [new Ri([new ga("", null)])]),
			this.first = n,
			this.scrollTop = this.scrollLeft = 0,
			this.cantEdit = !1,
			this.cleanGeneration = 1,
			this.frontier = n;
			var r = $o(n, 0);
			this.sel = he(r),
			this.history = new nr(null),
			this.id = ++ba,
			this.modeOption = t,
			this.lineSep = i,
			this.extend = !1,
			"string" == typeof e && (e = this.splitLines(e)),
			Yi(this, {
				from: r,
				to: r,
				text: e
			}),
			Se(this, he(r), Ea)
		};
		wa.prototype = Er(Ui.prototype, {
			constructor: wa,
			iter: function(e, t, n) {
				n ? this.iterN(e - this.first, t - e, n) : this.iterN(this.first, this.first + this.size, e)
			},
			insert: function(e, t) {
				for (var n = 0,
				i = 0; i < t.length; ++i) n += t[i].height;
				this.insertInner(e - this.first, t, n)
			},
			remove: function(e, t) {
				this.removeInner(e - this.first, t)
			},
			getValue: function(e) {
				var t = Zi(this, this.first, this.first + this.size);
				return ! 1 === e ? t: t.join(e || this.lineSeparator())
			},
			setValue: Nt(function(e) {
				var t = $o(this.first, 0),
				n = this.first + this.size - 1;
				Cn(this, {
					from: t,
					to: $o(n, Vi(this, n).text.length),
					text: this.splitLines(e),
					origin: "setValue",
					full: !0
				},
				!0),
				Se(this, he(t))
			}),
			replaceRange: function(e, t, n, i) {
				On(this, e, t = pe(this, t), n = n ? pe(this, n) : t, i)
			},
			getRange: function(e, t, n) {
				var i = Ki(this, pe(this, e), pe(this, t));
				return ! 1 === n ? i: i.join(n || this.lineSeparator())
			},
			getLine: function(e) {
				var t = this.getLineHandle(e);
				return t && t.text
			},
			getLineHandle: function(e) {
				return ge(this, e) ? Vi(this, e) : void 0
			},
			getLineNumber: function(e) {
				return Qi(e)
			},
			getLineHandleVisualStart: function(e) {
				return "number" == typeof e && (e = Vi(this, e)),
				gi(e)
			},
			lineCount: function() {
				return this.size
			},
			firstLine: function() {
				return this.first
			},
			lastLine: function() {
				return this.first + this.size - 1
			},
			clipPos: function(e) {
				return pe(this, e)
			},
			getCursor: function(e) {
				var t = this.sel.primary();
				return null == e || "head" == e ? t.head: "anchor" == e ? t.anchor: "end" == e || "to" == e || !1 === e ? t.to() : t.from()
			},
			listSelections: function() {
				return this.sel.ranges
			},
			somethingSelected: function() {
				return this.sel.somethingSelected()
			},
			setCursor: Nt(function(e, t, n) {
				xe(this, pe(this, "number" == typeof e ? $o(e, t || 0) : e), null, n)
			}),
			setSelection: Nt(function(e, t, n) {
				xe(this, pe(this, e), pe(this, t || e), n)
			}),
			extendSelection: Nt(function(e, t, n) {
				be(this, pe(this, e), t && pe(this, t), n)
			}),
			extendSelections: Nt(function(e, t) {
				we(this, ve(this, e), t)
			}),
			extendSelectionsBy: Nt(function(e, t) {
				we(this, ve(this, Nr(this.sel.ranges, e)), t)
			}),
			setSelections: Nt(function(e, t, n) {
				if (e.length) {
					for (var i = 0,
					r = []; i < e.length; i++) r[i] = new ue(pe(this, e[i].anchor), pe(this, e[i].head));
					null == t && (t = Math.min(e.length - 1, this.sel.primIndex)),
					Se(this, de(r, t), n)
				}
			}),
			addSelection: Nt(function(e, t, n) {
				var i = this.sel.ranges.slice(0);
				i.push(new ue(pe(this, e), pe(this, t || e))),
				Se(this, de(i, i.length - 1), n)
			}),
			getSelection: function(e) {
				for (var t, n = this.sel.ranges,
				i = 0; i < n.length; i++) {
					var r = Ki(this, n[i].from(), n[i].to());
					t = t ? t.concat(r) : r
				}
				return ! 1 === e ? t: t.join(e || this.lineSeparator())
			},
			getSelections: function(e) {
				for (var t = [], n = this.sel.ranges, i = 0; i < n.length; i++) {
					var r = Ki(this, n[i].from(), n[i].to()); ! 1 !== e && (r = r.join(e || this.lineSeparator())),
					t[i] = r
				}
				return t
			},
			replaceSelection: function(e, t, n) {
				for (var i = [], r = 0; r < this.sel.ranges.length; r++) i[r] = e;
				this.replaceSelections(i, t, n || "+input")
			},
			replaceSelections: Nt(function(e, t, n) {
				for (var i = [], r = this.sel, o = 0; o < r.ranges.length; o++) {
					var a = r.ranges[o];
					i[o] = {
						from: a.from(),
						to: a.to(),
						text: this.splitLines(e[o]),
						origin: n
					}
				}
				for (var s = t && "end" != t && xn(this, i, t), o = i.length - 1; o >= 0; o--) Cn(this, i[o]);
				s ? Ce(this, s) : this.cm && jn(this.cm)
			}),
			undo: Nt(function() {
				Mn(this, "undo")
			}),
			redo: Nt(function() {
				Mn(this, "redo")
			}),
			undoSelection: Nt(function() {
				Mn(this, "undo", !0)
			}),
			redoSelection: Nt(function() {
				Mn(this, "redo", !0)
			}),
			setExtending: function(e) {
				this.extend = e
			},
			getExtending: function() {
				return this.extend
			},
			historySize: function() {
				for (var e = this.history,
				t = 0,
				n = 0,
				i = 0; i < e.done.length; i++) e.done[i].ranges || ++t;
				for (i = 0; i < e.undone.length; i++) e.undone[i].ranges || ++n;
				return {
					undo: t,
					redo: n
				}
			},
			clearHistory: function() {
				this.history = new nr(this.history.maxGeneration)
			},
			markClean: function() {
				this.cleanGeneration = this.changeGeneration(!0)
			},
			changeGeneration: function(e) {
				return e && (this.history.lastOp = this.history.lastSelOp = this.history.lastOrigin = null),
				this.history.generation
			},
			isClean: function(e) {
				return this.history.generation == (e || this.cleanGeneration)
			},
			getHistory: function() {
				return {
					done: fr(this.history.done),
					undone: fr(this.history.undone)
				}
			},
			setHistory: function(e) {
				var t = this.history = new nr(this.history.maxGeneration);
				t.done = fr(e.done.slice(0), null, !0),
				t.undone = fr(e.undone.slice(0), null, !0)
			},
			addLineClass: Nt(function(e, t, n) {
				return zn(this, e, "gutter" == t ? "gutter": "class",
				function(e) {
					var i = "text" == t ? "textClass": "background" == t ? "bgClass": "gutter" == t ? "gutterClass": "wrapClass";
					if (e[i]) {
						if (Yr(n).test(e[i])) return ! 1;
						e[i] += " " + n
					} else e[i] = n;
					return ! 0
				})
			}),
			removeLineClass: Nt(function(e, t, n) {
				return zn(this, e, "gutter" == t ? "gutter": "class",
				function(e) {
					var i = "text" == t ? "textClass": "background" == t ? "bgClass": "gutter" == t ? "gutterClass": "wrapClass",
					r = e[i];
					if (!r) return ! 1;
					if (null == n) e[i] = null;
					else {
						var o = r.match(Yr(n));
						if (!o) return ! 1;
						var a = o.index + o[0].length;
						e[i] = r.slice(0, o.index) + (o.index && a != r.length ? " ": "") + r.slice(a) || null
					}
					return ! 0
				})
			}),
			addLineWidget: Nt(function(e, t, n) {
				return Ci(this, e, t, n)
			}),
			removeLineWidget: function(e) {
				e.clear()
			},
			markText: function(e, t, n) {
				return Bn(this, pe(this, e), pe(this, t), n, n && n.type || "range")
			},
			setBookmark: function(e, t) {
				var n = {
					replacedWith: t && (null == t.nodeType ? t.widget: t),
					insertLeft: t && t.insertLeft,
					clearWhenEmpty: !1,
					shared: t && t.shared,
					handleMouseEvents: t && t.handleMouseEvents
				};
				return e = pe(this, e),
				Bn(this, e, e, n, "bookmark")
			},
			findMarksAt: function(e) {
				var t = [],
				n = Vi(this, (e = pe(this, e)).line).markedSpans;
				if (n) for (var i = 0; i < n.length; ++i) {
					var r = n[i]; (null == r.from || r.from <= e.ch) && (null == r.to || r.to >= e.ch) && t.push(r.marker.parent || r.marker)
				}
				return t
			},
			findMarks: function(e, t, n) {
				e = pe(this, e),
				t = pe(this, t);
				var i = [],
				r = e.line;
				return this.iter(e.line, t.line + 1,
				function(o) {
					var a = o.markedSpans;
					if (a) for (var s = 0; s < a.length; s++) {
						var l = a[s];
						null != l.to && r == e.line && e.ch > l.to || null == l.from && r != e.line || null != l.from && r == t.line && l.from > t.ch || n && !n(l.marker) || i.push(l.marker.parent || l.marker)
					}++r
				}),
				i
			},
			getAllMarks: function() {
				var e = [];
				return this.iter(function(t) {
					var n = t.markedSpans;
					if (n) for (var i = 0; i < n.length; ++i) null != n[i].from && e.push(n[i].marker)
				}),
				e
			},
			posFromIndex: function(e) {
				var t, n = this.first,
				i = this.lineSeparator().length;
				return this.iter(function(r) {
					var o = r.text.length + i;
					return o > e ? (t = e, !0) : (e -= o, void++n)
				}),
				pe(this, $o(n, t))
			},
			indexFromPos: function(e) {
				var t = (e = pe(this, e)).ch;
				if (e.line < this.first || e.ch < 0) return 0;
				var n = this.lineSeparator().length;
				return this.iter(this.first, e.line,
				function(e) {
					t += e.text.length + n
				}),
				t
			},
			copy: function(e) {
				var t = new wa(Zi(this, this.first, this.first + this.size), this.modeOption, this.first, this.lineSep);
				return t.scrollTop = this.scrollTop,
				t.scrollLeft = this.scrollLeft,
				t.sel = this.sel,
				t.extend = !1,
				e && (t.history.undoDepth = this.history.undoDepth, t.setHistory(this.getHistory())),
				t
			},
			linkedDoc: function(e) {
				e || (e = {});
				var t = this.first,
				n = this.first + this.size;
				null != e.from && e.from > t && (t = e.from),
				null != e.to && e.to < n && (n = e.to);
				var i = new wa(Zi(this, t, n), e.mode || this.modeOption, t, this.lineSep);
				return e.sharedHist && (i.history = this.history),
				(this.linked || (this.linked = [])).push({
					doc: i,
					sharedHist: e.sharedHist
				}),
				i.linked = [{
					doc: this,
					isParent: !0,
					sharedHist: e.sharedHist
				}],
				Kn(i, Vn(this)),
				i
			},
			unlinkDoc: function(t) {
				if (t instanceof e && (t = t.doc), this.linked) for (var n = 0; n < this.linked.length; ++n) if (this.linked[n].doc == t) {
					this.linked.splice(n, 1),
					t.unlinkDoc(this),
					Zn(Vn(this));
					break
				}
				if (t.history == this.history) {
					var i = [t.id];
					Bi(t,
					function(e) {
						i.push(e.id)
					},
					!0),
					t.history = new nr(null),
					t.history.done = fr(this.history.done, i),
					t.history.undone = fr(this.history.undone, i)
				}
			},
			iterLinkedDocs: function(e) {
				Bi(this, e)
			},
			getMode: function() {
				return this.mode
			},
			getEditor: function() {
				return this.cm
			},
			splitLines: function(e) {
				return this.lineSep ? e.split(this.lineSep) : Xa(e)
			},
			lineSeparator: function() {
				return this.lineSep || "\n"
			}
		}),
		wa.prototype.eachLine = wa.prototype.iter;
		var ka = "iter insert remove copy getEditor constructor".split(" ");
		for (var xa in wa.prototype) wa.prototype.hasOwnProperty(xa) && Or(ka, xa) < 0 && (e.prototype[xa] = function(e) {
			return function() {
				return e.apply(this.doc, arguments)
			}
		} (wa.prototype[xa]));
		Mr(wa);
		var _a = e.e_preventDefault = function(e) {
			e.preventDefault ? e.preventDefault() : e.returnValue = !1
		},
		Ca = e.e_stopPropagation = function(e) {
			e.stopPropagation ? e.stopPropagation() : e.cancelBubble = !0
		},
		Sa = e.e_stop = function(e) {
			_a(e),
			Ca(e)
		},
		Ma = e.on = function(e, t, n) {
			if (e.addEventListener) e.addEventListener(t, n, !1);
			else if (e.attachEvent) e.attachEvent("on" + t, n);
			else {
				var i = e._handlers || (e._handlers = {}); (i[t] || (i[t] = [])).push(n)
			}
		},
		Ta = [],
		Da = e.off = function(e, t, n) {
			if (e.removeEventListener) e.removeEventListener(t, n, !1);
			else if (e.detachEvent) e.detachEvent("on" + t, n);
			else for (var i = wr(e, t, !1), r = 0; r < i.length; ++r) if (i[r] == n) {
				i.splice(r, 1);
				break
			}
		},
		La = e.signal = function(e, t) {
			var n = wr(e, t, !0);
			if (n.length) for (var i = Array.prototype.slice.call(arguments, 2), r = 0; r < n.length; ++r) n[r].apply(null, i)
		},
		Oa = null,
		Na = 30,
		Aa = e.Pass = {
			toString: function() {
				return "CodeMirror.Pass"
			}
		},
		Ea = {
			scroll: !1
		},
		$a = {
			origin: "*mouse"
		},
		qa = {
			origin: "+move"
		};
		Tr.prototype.set = function(e, t) {
			clearTimeout(this.id),
			this.id = setTimeout(t, e)
		};
		var ja = e.countColumn = function(e, t, n, i, r) {
			null == t && -1 == (t = e.search(/[^\s\u00a0]/)) && (t = e.length);
			for (var o = i || 0,
			a = r || 0;;) {
				var s = e.indexOf("\t", o);
				if (0 > s || s >= t) return a + (t - o);
				a += s - o,
				a += n - a % n,
				o = s + 1
			}
		},
		Pa = e.findColumn = function(e, t, n) {
			for (var i = 0,
			r = 0;;) {
				var o = e.indexOf("\t", i); - 1 == o && (o = e.length);
				var a = o - i;
				if (o == e.length || r + a >= t) return i + Math.min(a, t - r);
				if (r += o - i, r += n - r % n, i = o + 1, r >= t) return i
			}
		},
		Ia = [""],
		za = function(e) {
			e.select()
		};
		So ? za = function(e) {
			e.selectionStart = 0,
			e.selectionEnd = e.value.length
		}: go && (za = function(e) {
			try {
				e.select()
			} catch(e) {}
		});
		var Wa, Ha = /[\u00df\u0587\u0590-\u05f4\u0600-\u06ff\u3040-\u309f\u30a0-\u30ff\u3400-\u4db5\u4e00-\u9fcc\uac00-\ud7af]/,
		Fa = e.isWordChar = function(e) {
			return /\w/.test(e) || e > "" && (e.toUpperCase() != e.toLowerCase() || Ha.test(e))
		},
		Ya = /[\u0300-\u036f\u0483-\u0489\u0591-\u05bd\u05bf\u05c1\u05c2\u05c4\u05c5\u05c7\u0610-\u061a\u064b-\u065e\u0670\u06d6-\u06dc\u06de-\u06e4\u06e7\u06e8\u06ea-\u06ed\u0711\u0730-\u074a\u07a6-\u07b0\u07eb-\u07f3\u0816-\u0819\u081b-\u0823\u0825-\u0827\u0829-\u082d\u0900-\u0902\u093c\u0941-\u0948\u094d\u0951-\u0955\u0962\u0963\u0981\u09bc\u09be\u09c1-\u09c4\u09cd\u09d7\u09e2\u09e3\u0a01\u0a02\u0a3c\u0a41\u0a42\u0a47\u0a48\u0a4b-\u0a4d\u0a51\u0a70\u0a71\u0a75\u0a81\u0a82\u0abc\u0ac1-\u0ac5\u0ac7\u0ac8\u0acd\u0ae2\u0ae3\u0b01\u0b3c\u0b3e\u0b3f\u0b41-\u0b44\u0b4d\u0b56\u0b57\u0b62\u0b63\u0b82\u0bbe\u0bc0\u0bcd\u0bd7\u0c3e-\u0c40\u0c46-\u0c48\u0c4a-\u0c4d\u0c55\u0c56\u0c62\u0c63\u0cbc\u0cbf\u0cc2\u0cc6\u0ccc\u0ccd\u0cd5\u0cd6\u0ce2\u0ce3\u0d3e\u0d41-\u0d44\u0d4d\u0d57\u0d62\u0d63\u0dca\u0dcf\u0dd2-\u0dd4\u0dd6\u0ddf\u0e31\u0e34-\u0e3a\u0e47-\u0e4e\u0eb1\u0eb4-\u0eb9\u0ebb\u0ebc\u0ec8-\u0ecd\u0f18\u0f19\u0f35\u0f37\u0f39\u0f71-\u0f7e\u0f80-\u0f84\u0f86\u0f87\u0f90-\u0f97\u0f99-\u0fbc\u0fc6\u102d-\u1030\u1032-\u1037\u1039\u103a\u103d\u103e\u1058\u1059\u105e-\u1060\u1071-\u1074\u1082\u1085\u1086\u108d\u109d\u135f\u1712-\u1714\u1732-\u1734\u1752\u1753\u1772\u1773\u17b7-\u17bd\u17c6\u17c9-\u17d3\u17dd\u180b-\u180d\u18a9\u1920-\u1922\u1927\u1928\u1932\u1939-\u193b\u1a17\u1a18\u1a56\u1a58-\u1a5e\u1a60\u1a62\u1a65-\u1a6c\u1a73-\u1a7c\u1a7f\u1b00-\u1b03\u1b34\u1b36-\u1b3a\u1b3c\u1b42\u1b6b-\u1b73\u1b80\u1b81\u1ba2-\u1ba5\u1ba8\u1ba9\u1c2c-\u1c33\u1c36\u1c37\u1cd0-\u1cd2\u1cd4-\u1ce0\u1ce2-\u1ce8\u1ced\u1dc0-\u1de6\u1dfd-\u1dff\u200c\u200d\u20d0-\u20f0\u2cef-\u2cf1\u2de0-\u2dff\u302a-\u302f\u3099\u309a\ua66f-\ua672\ua67c\ua67d\ua6f0\ua6f1\ua802\ua806\ua80b\ua825\ua826\ua8c4\ua8e0-\ua8f1\ua926-\ua92d\ua947-\ua951\ua980-\ua982\ua9b3\ua9b6-\ua9b9\ua9bc\uaa29-\uaa2e\uaa31\uaa32\uaa35\uaa36\uaa43\uaa4c\uaab0\uaab2-\uaab4\uaab7\uaab8\uaabe\uaabf\uaac1\uabe5\uabe8\uabed\udc00-\udfff\ufb1e\ufe00-\ufe0f\ufe20-\ufe26\uff9e\uff9f]/;
		Wa = document.createRange ?
		function(e, t, n, i) {
			var r = document.createRange();
			return r.setEnd(i || e, n),
			r.setStart(e, t),
			r
		}: function(e, t, n) {
			var i = document.body.createTextRange();
			try {
				i.moveToElementText(e.parentNode)
			} catch(e) {
				return i
			}
			return i.collapse(!0),
			i.moveEnd("character", n),
			i.moveStart("character", t),
			i
		};
		var Ra = e.contains = function(e, t) {
			if (3 == t.nodeType && (t = t.parentNode), e.contains) return e.contains(t);
			do {
				if (11 == t.nodeType && (t = t.host), t == e) return ! 0
			} while ( t = t . parentNode )
		};
		go && 11 > vo && (Fr = function() {
			try {
				return document.activeElement
			} catch(e) {
				return document.body
			}
		});
		var Ua, Ba, Ga = e.rmClass = function(e, t) {
			var n = e.className,
			i = Yr(t).exec(n);
			if (i) {
				var r = n.slice(i.index + i[0].length);
				e.className = n.slice(0, i.index) + (r ? i[1] + r: "")
			}
		},
		Va = e.addClass = function(e, t) {
			var n = e.className;
			Yr(t).test(n) || (e.className += (n ? " ": "") + t)
		},
		Ka = !1,
		Za = function() {
			if (go && 9 > vo) return ! 1;
			var e = zr("div");
			return "draggable" in e || "dragDrop" in e
		} (),
		Xa = e.splitLines = 3 != "\n\nb".split(/\n/).length ?
		function(e) {
			for (var t = 0,
			n = [], i = e.length; i >= t;) {
				var r = e.indexOf("\n", t); - 1 == r && (r = e.length);
				var o = e.slice(t, "\r" == e.charAt(r - 1) ? r - 1 : r),
				a = o.indexOf("\r"); - 1 != a ? (n.push(o.slice(0, a)), t += a + 1) : (n.push(o), t = r + 1)
			}
			return n
		}: function(e) {
			return e.split(/\r\n?|\n/)
		},
		Qa = window.getSelection ?
		function(e) {
			try {
				return e.selectionStart != e.selectionEnd
			} catch(e) {
				return ! 1
			}
		}: function(e) {
			try {
				var t = e.ownerDocument.selection.createRange()
			} catch(e) {}
			return ! (!t || t.parentElement() != e) && 0 != t.compareEndPoints("StartToEnd", t)
		},
		Ja = function() {
			var e = zr("div");
			return "oncopy" in e || (e.setAttribute("oncopy", "return;"), "function" == typeof e.oncopy)
		} (),
		es = null,
		ts = e.keyNames = {
			3 : "Enter",
			8 : "Backspace",
			9 : "Tab",
			13 : "Enter",
			16 : "Shift",
			17 : "Ctrl",
			18 : "Alt",
			19 : "Pause",
			20 : "CapsLock",
			27 : "Esc",
			32 : "Space",
			33 : "PageUp",
			34 : "PageDown",
			35 : "End",
			36 : "Home",
			37 : "Left",
			38 : "Up",
			39 : "Right",
			40 : "Down",
			44 : "PrintScrn",
			45 : "Insert",
			46 : "Delete",
			59 : ";",
			61 : "=",
			91 : "Mod",
			92 : "Mod",
			93 : "Mod",
			106 : "*",
			107 : "=",
			109 : "-",
			110 : ".",
			111 : "/",
			127 : "Delete",
			173 : "-",
			186 : ";",
			187 : "=",
			188 : ",",
			189 : "-",
			190 : ".",
			191 : "/",
			192 : "`",
			219 : "[",
			220 : "\\",
			221 : "]",
			222 : "'",
			63232 : "Up",
			63233 : "Down",
			63234 : "Left",
			63235 : "Right",
			63272 : "Delete",
			63273 : "Home",
			63275 : "End",
			63276 : "PageUp",
			63277 : "PageDown",
			63302 : "Insert"
		}; !
		function() {
			for (e = 0; 10 > e; e++) ts[e + 48] = ts[e + 96] = String(e);
			for (e = 65; 90 >= e; e++) ts[e] = String.fromCharCode(e);
			for (var e = 1; 12 >= e; e++) ts[e + 111] = ts[e + 63235] = "F" + e
		} ();
		var ns, is = function() {
			function e(e) {
				return 247 >= e ? n.charAt(e) : e >= 1424 && 1524 >= e ? "R": e >= 1536 && 1773 >= e ? i.charAt(e - 1536) : e >= 1774 && 2220 >= e ? "r": e >= 8192 && 8203 >= e ? "w": 8204 == e ? "b": "L"
			}
			function t(e, t, n) {
				this.level = e,
				this.from = t,
				this.to = n
			}
			var n = "bbbbbbbbbtstwsbbbbbbbbbbbbbbssstwNN%%%NNNNNN,N,N1111111111NNNNNNNLLLLLLLLLLLLLLLLLLLLLLLLLLNNNNNNLLLLLLLLLLLLLLLLLLLLLLLLLLNNNNbbbbbbsbbbbbbbbbbbbbbbbbbbbbbbbbb,N%%%%NNNNLNNNNN%%11NLNNN1LNNNNNLLLLLLLLLLLLLLLLLLLLLLLNLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLN",
			i = "rrrrrrrrrrrr,rNNmmmmmmrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrmmmmmmmmmmmmmmrrrrrrrnnnnnnnnnn%nnrrrmrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrmmmmmmmmmmmmmmmmmmmNmmmm",
			r = /[\u0590-\u05f4\u0600-\u06ff\u0700-\u08ac]/,
			o = /[stwN]/,
			a = /[LRr]/,
			s = /[Lb1n]/,
			l = /[1n]/,
			c = "L";
			return function(n) {
				if (!r.test(n)) return ! 1;
				for (var i = n.length,
				u = [], d = 0; i > d; ++d) u.push(g = e(n.charCodeAt(d)));
				for (var d = 0,
				h = c; i > d; ++d)"m" == (g = u[d]) ? u[d] = h: h = g;
				for (var d = 0,
				f = c; i > d; ++d)"1" == (g = u[d]) && "r" == f ? u[d] = "n": a.test(g) && (f = g, "r" == g && (u[d] = "R"));
				for (var d = 1,
				h = u[0]; i - 1 > d; ++d)"+" == (g = u[d]) && "1" == h && "1" == u[d + 1] ? u[d] = "1": "," != g || h != u[d + 1] || "1" != h && "n" != h || (u[d] = h),
				h = g;
				for (d = 0; i > d; ++d) if ("," == (g = u[d])) u[d] = "N";
				else if ("%" == g) {
					for (v = d + 1; i > v && "%" == u[v]; ++v);
					for (var p = d && "!" == u[d - 1] || i > v && "1" == u[v] ? "1": "N", m = d; v > m; ++m) u[m] = p;
					d = v - 1
				}
				for (var d = 0,
				f = c; i > d; ++d) {
					var g = u[d];
					"L" == f && "1" == g ? u[d] = "L": a.test(g) && (f = g)
				}
				for (d = 0; i > d; ++d) if (o.test(u[d])) {
					for (var v = d + 1; i > v && o.test(u[v]); ++v);
					for (var y = "L" == (d ? u[d - 1] : c), b = "L" == (i > v ? u[v] : c), p = y || b ? "L": "R", m = d; v > m; ++m) u[m] = p;
					d = v - 1
				}
				for (var w, k = [], d = 0; i > d;) if (s.test(u[d])) {
					var x = d;
					for (++d; i > d && s.test(u[d]); ++d);
					k.push(new t(0, x, d))
				} else {
					var _ = d,
					C = k.length;
					for (++d; i > d && "L" != u[d]; ++d);
					for (m = _; d > m;) if (l.test(u[m])) {
						m > _ && k.splice(C, 0, new t(1, _, m));
						var S = m;
						for (++m; d > m && l.test(u[m]); ++m);
						k.splice(C, 0, new t(2, S, m)),
						_ = m
					} else++m;
					d > _ && k.splice(C, 0, new t(1, _, d))
				}
				return 1 == k[0].level && (w = n.match(/^\s+/)) && (k[0].from = w[0].length, k.unshift(new t(0, 0, w[0].length))),
				1 == Lr(k).level && (w = n.match(/\s+$/)) && (Lr(k).to -= w[0].length, k.push(new t(0, i - w[0].length, i))),
				2 == k[0].level && k.unshift(new t(1, k[0].to, k[0].to)),
				k[0].level != Lr(k).level && k.push(new t(k[0].level, i, i)),
				k
			}
		} ();
		return e.version = "5.13.3",
		e
	}),
	function(e) {
		"object" == typeof exports && "object" == typeof module ? e(require("../../lib/codemirror")) : "function" == typeof define && define.amd ? define(["../../lib/codemirror"], e) : e(CodeMirror)
	} (function(e) {
		"use strict";
		function t(e) {
			for (var t = {},
			n = 0; n < e.length; ++n) t[e[n]] = !0;
			return t
		}
		function n(e, t) {
			for (var n, i = !1; null != (n = e.next());) {
				if (i && "/" == n) {
					t.tokenize = null;
					break
				}
				i = "*" == n
			}
			return ["comment", "comment"]
		}
		e.defineMode("css",
		function(t, n) {
			function i(e, t) {
				return p = t,
				e
			}
			function r(e, t) {
				var n = e.next();
				if (v[n]) {
					var r = v[n](e, t);
					if (!1 !== r) return r
				}
				return "@" == n ? (e.eatWhile(/[\w\\\-]/), i("def", e.current())) : "=" == n || ("~" == n || "|" == n) && e.eat("=") ? i(null, "compare") : '"' == n || "'" == n ? (t.tokenize = o(n), t.tokenize(e, t)) : "#" == n ? (e.eatWhile(/[\w\\\-]/), i("atom", "hash")) : "!" == n ? (e.match(/^\s*\w*/), i("keyword", "important")) : /\d/.test(n) || "." == n && e.eat(/\d/) ? (e.eatWhile(/[\w.%]/), i("number", "unit")) : "-" !== n ? /[,+>*\/]/.test(n) ? i(null, "select-op") : "." == n && e.match(/^-?[_a-z][_a-z0-9-]*/i) ? i("qualifier", "qualifier") : /[:;{}\[\]\(\)]/.test(n) ? i(null, n) : "u" == n && e.match(/rl(-prefix)?\(/) || "d" == n && e.match("omain(") || "r" == n && e.match("egexp(") ? (e.backUp(1), t.tokenize = a, i("property", "word")) : /[\w\\\-]/.test(n) ? (e.eatWhile(/[\w\\\-]/), i("property", "word")) : i(null, null) : /[\d.]/.test(e.peek()) ? (e.eatWhile(/[\w.%]/), i("number", "unit")) : e.match(/^-[\w\\\-]+/) ? (e.eatWhile(/[\w\\\-]/), e.match(/^\s*:/, !1) ? i("variable-2", "variable-definition") : i("variable-2", "variable")) : e.match(/^\w+-/) ? i("meta", "meta") : void 0
			}
			function o(e) {
				return function(t, n) {
					for (var r, o = !1; null != (r = t.next());) {
						if (r == e && !o) {
							")" == e && t.backUp(1);
							break
						}
						o = !o && "\\" == r
					}
					return (r == e || !o && ")" != e) && (n.tokenize = null),
					i("string", "string")
				}
			}
			function a(e, t) {
				return e.next(),
				e.match(/\s*[\"\')]/, !1) ? t.tokenize = null: t.tokenize = o(")"),
				i(null, "(")
			}
			function s(e, t, n) {
				this.type = e,
				this.indent = t,
				this.prev = n
			}
			function l(e, t, n, i) {
				return e.context = new s(n, t.indentation() + (!1 === i ? 0 : g), e.context),
				n
			}
			function c(e) {
				return e.context.prev && (e.context = e.context.prev),
				e.context.type
			}
			function u(e, t, n) {
				return O[n.context.type](e, t, n)
			}
			function d(e, t, n, i) {
				for (var r = i || 1; r > 0; r--) n.context = n.context.prev;
				return u(e, t, n)
			}
			function h(e) {
				var t = e.current().toLowerCase();
				m = T.hasOwnProperty(t) ? "atom": M.hasOwnProperty(t) ? "keyword": "variable"
			}
			var f = n.inline;
			n.propertyKeywords || (n = e.resolveMode("text/css"));
			var p, m, g = t.indentUnit,
			v = n.tokenHooks,
			y = n.documentTypes || {},
			b = n.mediaTypes || {},
			w = n.mediaFeatures || {},
			k = n.mediaValueKeywords || {},
			x = n.propertyKeywords || {},
			_ = n.nonStandardPropertyKeywords || {},
			C = n.fontProperties || {},
			S = n.counterDescriptors || {},
			M = n.colorKeywords || {},
			T = n.valueKeywords || {},
			D = n.allowNested,
			L = !0 === n.supportsAtComponent,
			O = {};
			return O.top = function(e, t, n) {
				if ("{" == e) return l(n, t, "block");
				if ("}" == e && n.context.prev) return c(n);
				if (L && /@component/.test(e)) return l(n, t, "atComponentBlock");
				if (/^@(-moz-)?document$/.test(e)) return l(n, t, "documentTypes");
				if (/^@(media|supports|(-moz-)?document|import)$/.test(e)) return l(n, t, "atBlock");
				if (/^@(font-face|counter-style)/.test(e)) return n.stateArg = e,
				"restricted_atBlock_before";
				if (/^@(-(moz|ms|o|webkit)-)?keyframes$/.test(e)) return "keyframes";
				if (e && "@" == e.charAt(0)) return l(n, t, "at");
				if ("hash" == e) m = "builtin";
				else if ("word" == e) m = "tag";
				else {
					if ("variable-definition" == e) return "maybeprop";
					if ("interpolation" == e) return l(n, t, "interpolation");
					if (":" == e) return "pseudo";
					if (D && "(" == e) return l(n, t, "parens")
				}
				return n.context.type
			},
			O.block = function(e, t, n) {
				if ("word" == e) {
					var i = t.current().toLowerCase();
					return x.hasOwnProperty(i) ? (m = "property", "maybeprop") : _.hasOwnProperty(i) ? (m = "string-2", "maybeprop") : D ? (m = t.match(/^\s*:(?:\s|$)/, !1) ? "property": "tag", "block") : (m += " error", "maybeprop")
				}
				return "meta" == e ? "block": D || "hash" != e && "qualifier" != e ? O.top(e, t, n) : (m = "error", "block")
			},
			O.maybeprop = function(e, t, n) {
				return ":" == e ? l(n, t, "prop") : u(e, t, n)
			},
			O.prop = function(e, t, n) {
				if (";" == e) return c(n);
				if ("{" == e && D) return l(n, t, "propBlock");
				if ("}" == e || "{" == e) return d(e, t, n);
				if ("(" == e) return l(n, t, "parens");
				if ("hash" != e || /^#([0-9a-fA-f]{3,4}|[0-9a-fA-f]{6}|[0-9a-fA-f]{8})$/.test(t.current())) {
					if ("word" == e) h(t);
					else if ("interpolation" == e) return l(n, t, "interpolation")
				} else m += " error";
				return "prop"
			},
			O.propBlock = function(e, t, n) {
				return "}" == e ? c(n) : "word" == e ? (m = "property", "maybeprop") : n.context.type
			},
			O.parens = function(e, t, n) {
				return "{" == e || "}" == e ? d(e, t, n) : ")" == e ? c(n) : "(" == e ? l(n, t, "parens") : "interpolation" == e ? l(n, t, "interpolation") : ("word" == e && h(t), "parens")
			},
			O.pseudo = function(e, t, n) {
				return "word" == e ? (m = "variable-3", n.context.type) : u(e, t, n)
			},
			O.documentTypes = function(e, t, n) {
				return "word" == e && y.hasOwnProperty(t.current()) ? (m = "tag", n.context.type) : O.atBlock(e, t, n)
			},
			O.atBlock = function(e, t, n) {
				if ("(" == e) return l(n, t, "atBlock_parens");
				if ("}" == e || ";" == e) return d(e, t, n);
				if ("{" == e) return c(n) && l(n, t, D ? "block": "top");
				if ("interpolation" == e) return l(n, t, "interpolation");
				if ("word" == e) {
					var i = t.current().toLowerCase();
					m = "only" == i || "not" == i || "and" == i || "or" == i ? "keyword": b.hasOwnProperty(i) ? "attribute": w.hasOwnProperty(i) ? "property": k.hasOwnProperty(i) ? "keyword": x.hasOwnProperty(i) ? "property": _.hasOwnProperty(i) ? "string-2": T.hasOwnProperty(i) ? "atom": M.hasOwnProperty(i) ? "keyword": "error"
				}
				return n.context.type
			},
			O.atComponentBlock = function(e, t, n) {
				return "}" == e ? d(e, t, n) : "{" == e ? c(n) && l(n, t, D ? "block": "top", !1) : ("word" == e && (m = "error"), n.context.type)
			},
			O.atBlock_parens = function(e, t, n) {
				return ")" == e ? c(n) : "{" == e || "}" == e ? d(e, t, n, 2) : O.atBlock(e, t, n)
			},
			O.restricted_atBlock_before = function(e, t, n) {
				return "{" == e ? l(n, t, "restricted_atBlock") : "word" == e && "@counter-style" == n.stateArg ? (m = "variable", "restricted_atBlock_before") : u(e, t, n)
			},
			O.restricted_atBlock = function(e, t, n) {
				return "}" == e ? (n.stateArg = null, c(n)) : "word" == e ? (m = "@font-face" == n.stateArg && !C.hasOwnProperty(t.current().toLowerCase()) || "@counter-style" == n.stateArg && !S.hasOwnProperty(t.current().toLowerCase()) ? "error": "property", "maybeprop") : "restricted_atBlock"
			},
			O.keyframes = function(e, t, n) {
				return "word" == e ? (m = "variable", "keyframes") : "{" == e ? l(n, t, "top") : u(e, t, n)
			},
			O.at = function(e, t, n) {
				return ";" == e ? c(n) : "{" == e || "}" == e ? d(e, t, n) : ("word" == e ? m = "tag": "hash" == e && (m = "builtin"), "at")
			},
			O.interpolation = function(e, t, n) {
				return "}" == e ? c(n) : "{" == e || ";" == e ? d(e, t, n) : ("word" == e ? m = "variable": "variable" != e && "(" != e && ")" != e && (m = "error"), "interpolation")
			},
			{
				startState: function(e) {
					return {
						tokenize: null,
						state: f ? "block": "top",
						stateArg: null,
						context: new s(f ? "block": "top", e || 0, null)
					}
				},
				token: function(e, t) {
					if (!t.tokenize && e.eatSpace()) return null;
					var n = (t.tokenize || r)(e, t);
					return n && "object" == typeof n && (p = n[1], n = n[0]),
					m = n,
					t.state = O[t.state](p, e, t),
					m
				},
				indent: function(e, t) {
					var n = e.context,
					i = t && t.charAt(0),
					r = n.indent;
					return "prop" != n.type || "}" != i && ")" != i || (n = n.prev),
					n.prev && ("}" != i || "block" != n.type && "top" != n.type && "interpolation" != n.type && "restricted_atBlock" != n.type ? (")" == i && ("parens" == n.type || "atBlock_parens" == n.type) || "{" == i && ("at" == n.type || "atBlock" == n.type)) && (r = Math.max(0, n.indent - g), n = n.prev) : (n = n.prev, r = n.indent)),
					r
				},
				electricChars: "}",
				blockCommentStart: "/*",
				blockCommentEnd: "*/",
				fold: "brace"
			}
		});
		var i = ["domain", "regexp", "url", "url-prefix"],
		r = t(i),
		o = ["all", "aural", "braille", "handheld", "print", "projection", "screen", "tty", "tv", "embossed"],
		a = t(o),
		s = ["width", "min-width", "max-width", "height", "min-height", "max-height", "device-width", "min-device-width", "max-device-width", "device-height", "min-device-height", "max-device-height", "aspect-ratio", "min-aspect-ratio", "max-aspect-ratio", "device-aspect-ratio", "min-device-aspect-ratio", "max-device-aspect-ratio", "color", "min-color", "max-color", "color-index", "min-color-index", "max-color-index", "monochrome", "min-monochrome", "max-monochrome", "resolution", "min-resolution", "max-resolution", "scan", "grid", "orientation", "device-pixel-ratio", "min-device-pixel-ratio", "max-device-pixel-ratio", "pointer", "any-pointer", "hover", "any-hover"],
		l = t(s),
		c = ["landscape", "portrait", "none", "coarse", "fine", "on-demand", "hover", "interlace", "progressive"],
		u = t(c),
		d = ["align-content", "align-items", "align-self", "alignment-adjust", "alignment-baseline", "anchor-point", "animation", "animation-delay", "animation-direction", "animation-duration", "animation-fill-mode", "animation-iteration-count", "animation-name", "animation-play-state", "animation-timing-function", "appearance", "azimuth", "backface-visibility", "background", "background-attachment", "background-blend-mode", "background-clip", "background-color", "background-image", "background-origin", "background-position", "background-repeat", "background-size", "baseline-shift", "binding", "bleed", "bookmark-label", "bookmark-level", "bookmark-state", "bookmark-target", "border", "border-bottom", "border-bottom-color", "border-bottom-left-radius", "border-bottom-right-radius", "border-bottom-style", "border-bottom-width", "border-collapse", "border-color", "border-image", "border-image-outset", "border-image-repeat", "border-image-slice", "border-image-source", "border-image-width", "border-left", "border-left-color", "border-left-style", "border-left-width", "border-radius", "border-right", "border-right-color", "border-right-style", "border-right-width", "border-spacing", "border-style", "border-top", "border-top-color", "border-top-left-radius", "border-top-right-radius", "border-top-style", "border-top-width", "border-width", "bottom", "box-decoration-break", "box-shadow", "box-sizing", "break-after", "break-before", "break-inside", "caption-side", "clear", "clip", "color", "color-profile", "column-count", "column-fill", "column-gap", "column-rule", "column-rule-color", "column-rule-style", "column-rule-width", "column-span", "column-width", "columns", "content", "counter-increment", "counter-reset", "crop", "cue", "cue-after", "cue-before", "cursor", "direction", "display", "dominant-baseline", "drop-initial-after-adjust", "drop-initial-after-align", "drop-initial-before-adjust", "drop-initial-before-align", "drop-initial-size", "drop-initial-value", "elevation", "empty-cells", "fit", "fit-position", "flex", "flex-basis", "flex-direction", "flex-flow", "flex-grow", "flex-shrink", "flex-wrap", "float", "float-offset", "flow-from", "flow-into", "font", "font-feature-settings", "font-family", "font-kerning", "font-language-override", "font-size", "font-size-adjust", "font-stretch", "font-style", "font-synthesis", "font-variant", "font-variant-alternates", "font-variant-caps", "font-variant-east-asian", "font-variant-ligatures", "font-variant-numeric", "font-variant-position", "font-weight", "grid", "grid-area", "grid-auto-columns", "grid-auto-flow", "grid-auto-position", "grid-auto-rows", "grid-column", "grid-column-end", "grid-column-start", "grid-row", "grid-row-end", "grid-row-start", "grid-template", "grid-template-areas", "grid-template-columns", "grid-template-rows", "hanging-punctuation", "height", "hyphens", "icon", "image-orientation", "image-rendering", "image-resolution", "inline-box-align", "justify-content", "left", "letter-spacing", "line-break", "line-height", "line-stacking", "line-stacking-ruby", "line-stacking-shift", "line-stacking-strategy", "list-style", "list-style-image", "list-style-position", "list-style-type", "margin", "margin-bottom", "margin-left", "margin-right", "margin-top", "marker-offset", "marks", "marquee-direction", "marquee-loop", "marquee-play-count", "marquee-speed", "marquee-style", "max-height", "max-width", "min-height", "min-width", "move-to", "nav-down", "nav-index", "nav-left", "nav-right", "nav-up", "object-fit", "object-position", "opacity", "order", "orphans", "outline", "outline-color", "outline-offset", "outline-style", "outline-width", "overflow", "overflow-style", "overflow-wrap", "overflow-x", "overflow-y", "padding", "padding-bottom", "padding-left", "padding-right", "padding-top", "page", "page-break-after", "page-break-before", "page-break-inside", "page-policy", "pause", "pause-after", "pause-before", "perspective", "perspective-origin", "pitch", "pitch-range", "play-during", "position", "presentation-level", "punctuation-trim", "quotes", "region-break-after", "region-break-before", "region-break-inside", "region-fragment", "rendering-intent", "resize", "rest", "rest-after", "rest-before", "richness", "right", "rotation", "rotation-point", "ruby-align", "ruby-overhang", "ruby-position", "ruby-span", "shape-image-threshold", "shape-inside", "shape-margin", "shape-outside", "size", "speak", "speak-as", "speak-header", "speak-numeral", "speak-punctuation", "speech-rate", "stress", "string-set", "tab-size", "table-layout", "target", "target-name", "target-new", "target-position", "text-align", "text-align-last", "text-decoration", "text-decoration-color", "text-decoration-line", "text-decoration-skip", "text-decoration-style", "text-emphasis", "text-emphasis-color", "text-emphasis-position", "text-emphasis-style", "text-height", "text-indent", "text-justify", "text-outline", "text-overflow", "text-shadow", "text-size-adjust", "text-space-collapse", "text-transform", "text-underline-position", "text-wrap", "top", "transform", "transform-origin", "transform-style", "transition", "transition-delay", "transition-duration", "transition-property", "transition-timing-function", "unicode-bidi", "vertical-align", "visibility", "voice-balance", "voice-duration", "voice-family", "voice-pitch", "voice-range", "voice-rate", "voice-stress", "voice-volume", "volume", "white-space", "widows", "width", "word-break", "word-spacing", "word-wrap", "z-index", "clip-path", "clip-rule", "mask", "enable-background", "filter", "flood-color", "flood-opacity", "lighting-color", "stop-color", "stop-opacity", "pointer-events", "color-interpolation", "color-interpolation-filters", "color-rendering", "fill", "fill-opacity", "fill-rule", "image-rendering", "marker", "marker-end", "marker-mid", "marker-start", "shape-rendering", "stroke", "stroke-dasharray", "stroke-dashoffset", "stroke-linecap", "stroke-linejoin", "stroke-miterlimit", "stroke-opacity", "stroke-width", "text-rendering", "baseline-shift", "dominant-baseline", "glyph-orientation-horizontal", "glyph-orientation-vertical", "text-anchor", "writing-mode"],
		h = t(d),
		f = ["scrollbar-arrow-color", "scrollbar-base-color", "scrollbar-dark-shadow-color", "scrollbar-face-color", "scrollbar-highlight-color", "scrollbar-shadow-color", "scrollbar-3d-light-color", "scrollbar-track-color", "shape-inside", "searchfield-cancel-button", "searchfield-decoration", "searchfield-results-button", "searchfield-results-decoration", "zoom"],
		p = t(f),
		m = t(["font-family", "src", "unicode-range", "font-variant", "font-feature-settings", "font-stretch", "font-weight", "font-style"]),
		g = t(["additive-symbols", "fallback", "negative", "pad", "prefix", "range", "speak-as", "suffix", "symbols", "system"]),
		v = ["aliceblue", "antiquewhite", "aqua", "aquamarine", "azure", "beige", "bisque", "black", "blanchedalmond", "blue", "blueviolet", "brown", "burlywood", "cadetblue", "chartreuse", "chocolate", "coral", "cornflowerblue", "cornsilk", "crimson", "cyan", "darkblue", "darkcyan", "darkgoldenrod", "darkgray", "darkgreen", "darkkhaki", "darkmagenta", "darkolivegreen", "darkorange", "darkorchid", "darkred", "darksalmon", "darkseagreen", "darkslateblue", "darkslategray", "darkturquoise", "darkviolet", "deeppink", "deepskyblue", "dimgray", "dodgerblue", "firebrick", "floralwhite", "forestgreen", "fuchsia", "gainsboro", "ghostwhite", "gold", "goldenrod", "gray", "grey", "green", "greenyellow", "honeydew", "hotpink", "indianred", "indigo", "ivory", "khaki", "lavender", "lavenderblush", "lawngreen", "lemonchiffon", "lightblue", "lightcoral", "lightcyan", "lightgoldenrodyellow", "lightgray", "lightgreen", "lightpink", "lightsalmon", "lightseagreen", "lightskyblue", "lightslategray", "lightsteelblue", "lightyellow", "lime", "limegreen", "linen", "magenta", "maroon", "mediumaquamarine", "mediumblue", "mediumorchid", "mediumpurple", "mediumseagreen", "mediumslateblue", "mediumspringgreen", "mediumturquoise", "mediumvioletred", "midnightblue", "mintcream", "mistyrose", "moccasin", "navajowhite", "navy", "oldlace", "olive", "olivedrab", "orange", "orangered", "orchid", "palegoldenrod", "palegreen", "paleturquoise", "palevioletred", "papayawhip", "peachpuff", "peru", "pink", "plum", "powderblue", "purple", "rebeccapurple", "red", "rosybrown", "royalblue", "saddlebrown", "salmon", "sandybrown", "seagreen", "seashell", "sienna", "silver", "skyblue", "slateblue", "slategray", "snow", "springgreen", "steelblue", "tan", "teal", "thistle", "tomato", "turquoise", "violet", "wheat", "white", "whitesmoke", "yellow", "yellowgreen"],
		y = t(v),
		b = ["above", "absolute", "activeborder", "additive", "activecaption", "afar", "after-white-space", "ahead", "alias", "all", "all-scroll", "alphabetic", "alternate", "always", "amharic", "amharic-abegede", "antialiased", "appworkspace", "arabic-indic", "armenian", "asterisks", "attr", "auto", "avoid", "avoid-column", "avoid-page", "avoid-region", "background", "backwards", "baseline", "below", "bidi-override", "binary", "bengali", "blink", "block", "block-axis", "bold", "bolder", "border", "border-box", "both", "bottom", "break", "break-all", "break-word", "bullets", "button", "button-bevel", "buttonface", "buttonhighlight", "buttonshadow", "buttontext", "calc", "cambodian", "capitalize", "caps-lock-indicator", "caption", "captiontext", "caret", "cell", "center", "checkbox", "circle", "cjk-decimal", "cjk-earthly-branch", "cjk-heavenly-stem", "cjk-ideographic", "clear", "clip", "close-quote", "col-resize", "collapse", "color", "color-burn", "color-dodge", "column", "column-reverse", "compact", "condensed", "contain", "content", "content-box", "context-menu", "continuous", "copy", "counter", "counters", "cover", "crop", "cross", "crosshair", "currentcolor", "cursive", "cyclic", "darken", "dashed", "decimal", "decimal-leading-zero", "default", "default-button", "destination-atop", "destination-in", "destination-out", "destination-over", "devanagari", "difference", "disc", "discard", "disclosure-closed", "disclosure-open", "document", "dot-dash", "dot-dot-dash", "dotted", "double", "down", "e-resize", "ease", "ease-in", "ease-in-out", "ease-out", "element", "ellipse", "ellipsis", "embed", "end", "ethiopic", "ethiopic-abegede", "ethiopic-abegede-am-et", "ethiopic-abegede-gez", "ethiopic-abegede-ti-er", "ethiopic-abegede-ti-et", "ethiopic-halehame-aa-er", "ethiopic-halehame-aa-et", "ethiopic-halehame-am-et", "ethiopic-halehame-gez", "ethiopic-halehame-om-et", "ethiopic-halehame-sid-et", "ethiopic-halehame-so-et", "ethiopic-halehame-ti-er", "ethiopic-halehame-ti-et", "ethiopic-halehame-tig", "ethiopic-numeric", "ew-resize", "exclusion", "expanded", "extends", "extra-condensed", "extra-expanded", "fantasy", "fast", "fill", "fixed", "flat", "flex", "flex-end", "flex-start", "footnotes", "forwards", "from", "geometricPrecision", "georgian", "graytext", "groove", "gujarati", "gurmukhi", "hand", "hangul", "hangul-consonant", "hard-light", "hebrew", "help", "hidden", "hide", "higher", "highlight", "highlighttext", "hiragana", "hiragana-iroha", "horizontal", "hsl", "hsla", "hue", "icon", "ignore", "inactiveborder", "inactivecaption", "inactivecaptiontext", "infinite", "infobackground", "infotext", "inherit", "initial", "inline", "inline-axis", "inline-block", "inline-flex", "inline-table", "inset", "inside", "intrinsic", "invert", "italic", "japanese-formal", "japanese-informal", "justify", "kannada", "katakana", "katakana-iroha", "keep-all", "khmer", "korean-hangul-formal", "korean-hanja-formal", "korean-hanja-informal", "landscape", "lao", "large", "larger", "left", "level", "lighter", "lighten", "line-through", "linear", "linear-gradient", "lines", "list-item", "listbox", "listitem", "local", "logical", "loud", "lower", "lower-alpha", "lower-armenian", "lower-greek", "lower-hexadecimal", "lower-latin", "lower-norwegian", "lower-roman", "lowercase", "ltr", "luminosity", "malayalam", "match", "matrix", "matrix3d", "media-controls-background", "media-current-time-display", "media-fullscreen-button", "media-mute-button", "media-play-button", "media-return-to-realtime-button", "media-rewind-button", "media-seek-back-button", "media-seek-forward-button", "media-slider", "media-sliderthumb", "media-time-remaining-display", "media-volume-slider", "media-volume-slider-container", "media-volume-sliderthumb", "medium", "menu", "menulist", "menulist-button", "menulist-text", "menulist-textfield", "menutext", "message-box", "middle", "min-intrinsic", "mix", "mongolian", "monospace", "move", "multiple", "multiply", "myanmar", "n-resize", "narrower", "ne-resize", "nesw-resize", "no-close-quote", "no-drop", "no-open-quote", "no-repeat", "none", "normal", "not-allowed", "nowrap", "ns-resize", "numbers", "numeric", "nw-resize", "nwse-resize", "oblique", "octal", "open-quote", "optimizeLegibility", "optimizeSpeed", "oriya", "oromo", "outset", "outside", "outside-shape", "overlay", "overline", "padding", "padding-box", "painted", "page", "paused", "persian", "perspective", "plus-darker", "plus-lighter", "pointer", "polygon", "portrait", "pre", "pre-line", "pre-wrap", "preserve-3d", "progress", "push-button", "radial-gradient", "radio", "read-only", "read-write", "read-write-plaintext-only", "rectangle", "region", "relative", "repeat", "repeating-linear-gradient", "repeating-radial-gradient", "repeat-x", "repeat-y", "reset", "reverse", "rgb", "rgba", "ridge", "right", "rotate", "rotate3d", "rotateX", "rotateY", "rotateZ", "round", "row", "row-resize", "row-reverse", "rtl", "run-in", "running", "s-resize", "sans-serif", "saturation", "scale", "scale3d", "scaleX", "scaleY", "scaleZ", "screen", "scroll", "scrollbar", "se-resize", "searchfield", "searchfield-cancel-button", "searchfield-decoration", "searchfield-results-button", "searchfield-results-decoration", "semi-condensed", "semi-expanded", "separate", "serif", "show", "sidama", "simp-chinese-formal", "simp-chinese-informal", "single", "skew", "skewX", "skewY", "skip-white-space", "slide", "slider-horizontal", "slider-vertical", "sliderthumb-horizontal", "sliderthumb-vertical", "slow", "small", "small-caps", "small-caption", "smaller", "soft-light", "solid", "somali", "source-atop", "source-in", "source-out", "source-over", "space", "space-around", "space-between", "spell-out", "square", "square-button", "start", "static", "status-bar", "stretch", "stroke", "sub", "subpixel-antialiased", "super", "sw-resize", "symbolic", "symbols", "table", "table-caption", "table-cell", "table-column", "table-column-group", "table-footer-group", "table-header-group", "table-row", "table-row-group", "tamil", "telugu", "text", "text-bottom", "text-top", "textarea", "textfield", "thai", "thick", "thin", "threeddarkshadow", "threedface", "threedhighlight", "threedlightshadow", "threedshadow", "tibetan", "tigre", "tigrinya-er", "tigrinya-er-abegede", "tigrinya-et", "tigrinya-et-abegede", "to", "top", "trad-chinese-formal", "trad-chinese-informal", "translate", "translate3d", "translateX", "translateY", "translateZ", "transparent", "ultra-condensed", "ultra-expanded", "underline", "up", "upper-alpha", "upper-armenian", "upper-greek", "upper-hexadecimal", "upper-latin", "upper-norwegian", "upper-roman", "uppercase", "urdu", "url", "var", "vertical", "vertical-text", "visible", "visibleFill", "visiblePainted", "visibleStroke", "visual", "w-resize", "wait", "wave", "wider", "window", "windowframe", "windowtext", "words", "wrap", "wrap-reverse", "x-large", "x-small", "xor", "xx-large", "xx-small"],
		w = t(b),
		k = i.concat(o).concat(s).concat(c).concat(d).concat(f).concat(v).concat(b);
		e.registerHelper("hintWords", "css", k),
		e.defineMIME("text/css", {
			documentTypes: r,
			mediaTypes: a,
			mediaFeatures: l,
			mediaValueKeywords: u,
			propertyKeywords: h,
			nonStandardPropertyKeywords: p,
			fontProperties: m,
			counterDescriptors: g,
			colorKeywords: y,
			valueKeywords: w,
			tokenHooks: {
				"/": function(e, t) {
					return !! e.eat("*") && (t.tokenize = n, n(e, t))
				}
			},
			name: "css"
		}),
		e.defineMIME("text/x-scss", {
			mediaTypes: a,
			mediaFeatures: l,
			mediaValueKeywords: u,
			propertyKeywords: h,
			nonStandardPropertyKeywords: p,
			colorKeywords: y,
			valueKeywords: w,
			fontProperties: m,
			allowNested: !0,
			tokenHooks: {
				"/": function(e, t) {
					return e.eat("/") ? (e.skipToEnd(), ["comment", "comment"]) : e.eat("*") ? (t.tokenize = n, n(e, t)) : ["operator", "operator"]
				},
				":": function(e) {
					return !! e.match(/\s*\{/) && [null, "{"]
				},
				$: function(e) {
					return e.match(/^[\w-]+/),
					e.match(/^\s*:/, !1) ? ["variable-2", "variable-definition"] : ["variable-2", "variable"]
				},
				"#": function(e) {
					return !! e.eat("{") && [null, "interpolation"]
				}
			},
			name: "css",
			helperType: "scss"
		}),
		e.defineMIME("text/x-less", {
			mediaTypes: a,
			mediaFeatures: l,
			mediaValueKeywords: u,
			propertyKeywords: h,
			nonStandardPropertyKeywords: p,
			colorKeywords: y,
			valueKeywords: w,
			fontProperties: m,
			allowNested: !0,
			tokenHooks: {
				"/": function(e, t) {
					return e.eat("/") ? (e.skipToEnd(), ["comment", "comment"]) : e.eat("*") ? (t.tokenize = n, n(e, t)) : ["operator", "operator"]
				},
				"@": function(e) {
					return e.eat("{") ? [null, "interpolation"] : !e.match(/^(charset|document|font-face|import|(-(moz|ms|o|webkit)-)?keyframes|media|namespace|page|supports)\b/, !1) && (e.eatWhile(/[\w\\\-]/), e.match(/^\s*:/, !1) ? ["variable-2", "variable-definition"] : ["variable-2", "variable"])
				},
				"&": function() {
					return ["atom", "atom"]
				}
			},
			name: "css",
			helperType: "less"
		}),
		e.defineMIME("text/x-gss", {
			documentTypes: r,
			mediaTypes: a,
			mediaFeatures: l,
			propertyKeywords: h,
			nonStandardPropertyKeywords: p,
			fontProperties: m,
			counterDescriptors: g,
			colorKeywords: y,
			valueKeywords: w,
			supportsAtComponent: !0,
			tokenHooks: {
				"/": function(e, t) {
					return !! e.eat("*") && (t.tokenize = n, n(e, t))
				}
			},
			name: "css",
			helperType: "gss"
		})
	}),
	function(e) {
		"object" == typeof exports && "object" == typeof module ? e(require("../../lib/codemirror"), require("../markdown/markdown"), require("../../addon/mode/overlay")) : "function" == typeof define && define.amd ? define(["../../lib/codemirror", "../markdown/markdown", "../../addon/mode/overlay"], e) : e(CodeMirror)
	} (function(e) {
		"use strict";
		var t = /^((?:(?:aaas?|about|acap|adiumxtra|af[ps]|aim|apt|attachment|aw|beshare|bitcoin|bolo|callto|cap|chrome(?:-extension)?|cid|coap|com-eventbrite-attendee|content|crid|cvs|data|dav|dict|dlna-(?:playcontainer|playsingle)|dns|doi|dtn|dvb|ed2k|facetime|feed|file|finger|fish|ftp|geo|gg|git|gizmoproject|go|gopher|gtalk|h323|hcp|https?|iax|icap|icon|im|imap|info|ipn|ipp|irc[6s]?|iris(?:\.beep|\.lwz|\.xpc|\.xpcs)?|itms|jar|javascript|jms|keyparc|lastfm|ldaps?|magnet|mailto|maps|market|message|mid|mms|ms-help|msnim|msrps?|mtqp|mumble|mupdate|mvn|news|nfs|nih?|nntp|notes|oid|opaquelocktoken|palm|paparazzi|platform|pop|pres|proxy|psyc|query|res(?:ource)?|rmi|rsync|rtmp|rtsp|secondlife|service|session|sftp|sgn|shttp|sieve|sips?|skype|sm[bs]|snmp|soap\.beeps?|soldat|spotify|ssh|steam|svn|tag|teamspeak|tel(?:net)?|tftp|things|thismessage|tip|tn3270|tv|udp|unreal|urn|ut2004|vemmi|ventrilo|view-source|webcal|wss?|wtai|wyciwyg|xcon(?:-userid)?|xfire|xmlrpc\.beeps?|xmpp|xri|ymsgr|z39\.50[rs]?):(?:\/{1,3}|[a-z0-9%])|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}\/)(?:[^\s()<>]|\([^\s()<>]*\))+(?:\([^\s()<>]*\)|[^\s`*!()\[\]{};:'".,<>?\xab\xbb\u201c\u201d\u2018\u2019]))/i;
		e.defineMode("gfm",
		function(n, i) {
			var r = 0,
			o = {
				startState: function() {
					return {
						code: !1,
						codeBlock: !1,
						ateSpace: !1
					}
				},
				copyState: function(e) {
					return {
						code: e.code,
						codeBlock: e.codeBlock,
						ateSpace: e.ateSpace
					}
				},
				token: function(e, n) {
					if (n.combineTokens = null, n.codeBlock) return e.match(/^```+/) ? (n.codeBlock = !1, null) : (e.skipToEnd(), null);
					if (e.sol() && (n.code = !1), e.sol() && e.match(/^```+/)) return e.skipToEnd(),
					n.codeBlock = !0,
					null;
					if ("`" === e.peek()) {
						e.next();
						var o = e.pos;
						e.eatWhile("`");
						var a = 1 + e.pos - o;
						return n.code ? a === r && (n.code = !1) : (r = a, n.code = !0),
						null
					}
					if (n.code) return e.next(),
					null;
					if (e.eatSpace()) return n.ateSpace = !0,
					null;
					if ((e.sol() || n.ateSpace) && (n.ateSpace = !1, !1 !== i.gitHubSpice)) {
						if (e.match(/^(?:[a-zA-Z0-9\-_]+\/)?(?:[a-zA-Z0-9\-_]+@)?(?:[a-f0-9]{7,40}\b)/)) return n.combineTokens = !0,
						"link";
						if (e.match(/^(?:[a-zA-Z0-9\-_]+\/)?(?:[a-zA-Z0-9\-_]+)?#[0-9]+\b/)) return n.combineTokens = !0,
						"link"
					}
					return e.match(t) && "](" != e.string.slice(e.start - 2, e.start) && (0 == e.start || /\W/.test(e.string.charAt(e.start - 1))) ? (n.combineTokens = !0, "link") : (e.next(), null)
				},
				blankLine: function(e) {
					return e.code = !1,
					null
				}
			},
			a = {
				underscoresBreakWords: !1,
				taskLists: !0,
				fencedCodeBlocks: "```",
				strikethrough: !0
			};
			for (var s in i) a[s] = i[s];
			return a.name = "markdown",
			e.overlayMode(e.getMode(n, a), o)
		},
		"markdown"),
		e.defineMIME("text/x-gfm", "gfm")
	}),
	function(e) {
		"object" == typeof exports && "object" == typeof module ? e(require("../../lib/codemirror"), require("../htmlmixed/htmlmixed"), require("../../addon/mode/multiplex")) : "function" == typeof define && define.amd ? define(["../../lib/codemirror", "../htmlmixed/htmlmixed", "../../addon/mode/multiplex"], e) : e(CodeMirror)
	} (function(e) {
		"use strict";
		e.defineMode("htmlembedded",
		function(t, n) {
			return e.multiplexingMode(e.getMode(t, "htmlmixed"), {
				open: n.open || n.scriptStartRegex || "<%",
				close: n.close || n.scriptEndRegex || "%>",
				mode: e.getMode(t, n.scriptingModeSpec)
			})
		},
		"htmlmixed"),
		e.defineMIME("application/x-ejs", {
			name: "htmlembedded",
			scriptingModeSpec: "javascript"
		}),
		e.defineMIME("application/x-aspx", {
			name: "htmlembedded",
			scriptingModeSpec: "text/x-csharp"
		}),
		e.defineMIME("application/x-jsp", {
			name: "htmlembedded",
			scriptingModeSpec: "text/x-java"
		}),
		e.defineMIME("application/x-erb", {
			name: "htmlembedded",
			scriptingModeSpec: "ruby"
		})
	}),
	function(e) {
		"object" == typeof exports && "object" == typeof module ? e(require("../../lib/codemirror"), require("../xml/xml"), require("../javascript/javascript"), require("../css/css")) : "function" == typeof define && define.amd ? define(["../../lib/codemirror", "../xml/xml", "../javascript/javascript", "../css/css"], e) : e(CodeMirror)
	} (function(e) {
		"use strict";
		function t(e, t, n) {
			var i = e.current(),
			r = i.search(t);
			return r > -1 ? e.backUp(i.length - r) : i.match(/<\/?$/) && (e.backUp(i.length), e.match(t, !1) || e.match(i)),
			n
		}
		function n(e) {
			var t = l[e];
			return t || (l[e] = new RegExp("\\s+" + e + "\\s*=\\s*('|\")?([^'\"]+)('|\")?\\s*"))
		}
		function i(e, t) {
			var i = e.match(n(t));
			return i ? i[2] : ""
		}
		function r(e, t) {
			return new RegExp((t ? "^": "") + "</s*" + e + "s*>", "i")
		}
		function o(e, t) {
			for (var n in e) for (var i = t[n] || (t[n] = []), r = e[n], o = r.length - 1; o >= 0; o--) i.unshift(r[o])
		}
		function a(e, t) {
			for (var n = 0; n < e.length; n++) {
				var r = e[n];
				if (!r[0] || r[1].test(i(t, r[0]))) return r[2]
			}
		}
		var s = {
			script: [["lang", /(javascript|babel)/i, "javascript"], ["type", /^(?:text|application)\/(?:x-)?(?:java|ecma)script$|^$/i, "javascript"], ["type", /./, "text/plain"], [null, null, "javascript"]],
			style: [["lang", /^css$/i, "css"], ["type", /^(text\/)?(x-)?(stylesheet|css)$/i, "css"], ["type", /./, "text/plain"], [null, null, "css"]]
		},
		l = {};
		e.defineMode("htmlmixed",
		function(n, i) {
			function l(i, o) {
				var s, d = c.token(i, o.htmlState),
				h = /\btag\b/.test(d);
				if (h && !/[<>\s\/]/.test(i.current()) && (s = o.htmlState.tagName && o.htmlState.tagName.toLowerCase()) && u.hasOwnProperty(s)) o.inTag = s + " ";
				else if (o.inTag && h && />$/.test(i.current())) {
					var f = /^([\S]+) (.*)/.exec(o.inTag);
					o.inTag = null;
					var p = ">" == i.current() && a(u[f[1]], f[2]),
					m = e.getMode(n, p),
					g = r(f[1], !0),
					v = r(f[1], !1);
					o.token = function(e, n) {
						return e.match(g, !1) ? (n.token = l, n.localState = n.localMode = null, null) : t(e, v, n.localMode.token(e, n.localState))
					},
					o.localMode = m,
					o.localState = e.startState(m, c.indent(o.htmlState, ""))
				} else o.inTag && (o.inTag += i.current(), i.eol() && (o.inTag += " "));
				return d
			}
			var c = e.getMode(n, {
				name: "xml",
				htmlMode: !0,
				multilineTagIndentFactor: i.multilineTagIndentFactor,
				multilineTagIndentPastTag: i.multilineTagIndentPastTag
			}),
			u = {},
			d = i && i.tags,
			h = i && i.scriptTypes;
			if (o(s, u), d && o(d, u), h) for (var f = h.length - 1; f >= 0; f--) u.script.unshift(["type", h[f].matches, h[f].mode]);
			return {
				startState: function() {
					return {
						token: l,
						inTag: null,
						localMode: null,
						localState: null,
						htmlState: c.startState()
					}
				},
				copyState: function(t) {
					var n;
					return t.localState && (n = e.copyState(t.localMode, t.localState)),
					{
						token: t.token,
						inTag: t.inTag,
						localMode: t.localMode,
						localState: n,
						htmlState: e.copyState(c, t.htmlState)
					}
				},
				token: function(e, t) {
					return t.token(e, t)
				},
				indent: function(t, n) {
					return ! t.localMode || /^\s*<\//.test(n) ? c.indent(t.htmlState, n) : t.localMode.indent ? t.localMode.indent(t.localState, n) : e.Pass
				},
				innerMode: function(e) {
					return {
						state: e.localState || e.htmlState,
						mode: e.localMode || c
					}
				}
			}
		},
		"xml", "javascript", "css"),
		e.defineMIME("text/html", "htmlmixed")
	}),
	function(e) {
		"object" == typeof exports && "object" == typeof module ? e(require("../../lib/codemirror")) : "function" == typeof define && define.amd ? define(["../../lib/codemirror"], e) : e(CodeMirror)
	} (function(e) {
		"use strict";
		function t(e, t, n) {
			return /^(?:operator|sof|keyword c|case|new|[\[{}\(,;:]|=>)$/.test(t.lastType) || "quasi" == t.lastType && /\{\s*$/.test(e.string.slice(0, e.pos - (n || 0)))
		}
		e.defineMode("javascript",
		function(n, i) {
			function r(e) {
				for (var t, n = !1,
				i = !1; null != (t = e.next());) {
					if (!n) {
						if ("/" == t && !i) return;
						"[" == t ? i = !0 : i && "]" == t && (i = !1)
					}
					n = !n && "\\" == t
				}
			}
			function o(e, t, n) {
				return we = e,
				ke = n,
				t
			}
			function a(e, n) {
				var i = e.next();
				if ('"' == i || "'" == i) return n.tokenize = s(i),
				n.tokenize(e, n);
				if ("." == i && e.match(/^\d+(?:[eE][+\-]?\d+)?/)) return o("number", "number");
				if ("." == i && e.match("..")) return o("spread", "meta");
				if (/[\[\]{}\(\),;\:\.]/.test(i)) return o(i);
				if ("=" == i && e.eat(">")) return o("=>", "operator");
				if ("0" == i && e.eat(/x/i)) return e.eatWhile(/[\da-f]/i),
				o("number", "number");
				if ("0" == i && e.eat(/o/i)) return e.eatWhile(/[0-7]/i),
				o("number", "number");
				if ("0" == i && e.eat(/b/i)) return e.eatWhile(/[01]/i),
				o("number", "number");
				if (/\d/.test(i)) return e.match(/^\d*(?:\.\d*)?(?:[eE][+\-]?\d+)?/),
				o("number", "number");
				if ("/" == i) return e.eat("*") ? (n.tokenize = l, l(e, n)) : e.eat("/") ? (e.skipToEnd(), o("comment", "comment")) : t(e, n, 1) ? (r(e), e.match(/^\b(([gimyu])(?![gimyu]*\2))+\b/), o("regexp", "string-2")) : (e.eatWhile(Le), o("operator", "operator", e.current()));
				if ("`" == i) return n.tokenize = c,
				c(e, n);
				if ("#" == i) return e.skipToEnd(),
				o("error", "error");
				if (Le.test(i)) return e.eatWhile(Le),
				o("operator", "operator", e.current());
				if (Te.test(i)) {
					e.eatWhile(Te);
					var a = e.current(),
					u = De.propertyIsEnumerable(a) && De[a];
					return u && "." != n.lastType ? o(u.type, u.style, a) : o("variable", "variable", a)
				}
			}
			function s(e) {
				return function(t, n) {
					var i, r = !1;
					if (Ce && "@" == t.peek() && t.match(Oe)) return n.tokenize = a,
					o("jsonld-keyword", "meta");
					for (; null != (i = t.next()) && (i != e || r);) r = !r && "\\" == i;
					return r || (n.tokenize = a),
					o("string", "string")
				}
			}
			function l(e, t) {
				for (var n, i = !1; n = e.next();) {
					if ("/" == n && i) {
						t.tokenize = a;
						break
					}
					i = "*" == n
				}
				return o("comment", "comment")
			}
			function c(e, t) {
				for (var n, i = !1; null != (n = e.next());) {
					if (!i && ("`" == n || "$" == n && e.eat("{"))) {
						t.tokenize = a;
						break
					}
					i = !i && "\\" == n
				}
				return o("quasi", "string-2", e.current())
			}
			function u(e, t) {
				t.fatArrowAt && (t.fatArrowAt = null);
				var n = e.string.indexOf("=>", e.start);
				if (! (0 > n)) {
					for (var i = 0,
					r = !1,
					o = n - 1; o >= 0; --o) {
						var a = e.string.charAt(o),
						s = Ne.indexOf(a);
						if (s >= 0 && 3 > s) {
							if (!i) {++o;
								break
							}
							if (0 == --i) break
						} else if (s >= 3 && 6 > s)++i;
						else if (Te.test(a)) r = !0;
						else {
							if (/["'\/]/.test(a)) return;
							if (r && !i) {++o;
								break
							}
						}
					}
					r && !i && (t.fatArrowAt = o)
				}
			}
			function d(e, t, n, i, r, o) {
				this.indented = e,
				this.column = t,
				this.type = n,
				this.prev = r,
				this.info = o,
				null != i && (this.align = i)
			}
			function h(e, t) {
				for (i = e.localVars; i; i = i.next) if (i.name == t) return ! 0;
				for (var n = e.context; n; n = n.prev) for (var i = n.vars; i; i = i.next) if (i.name == t) return ! 0
			}
			function f(e, t, n, i, r) {
				var o = e.cc;
				for (Ee.state = e, Ee.stream = r, Ee.marked = null, Ee.cc = o, Ee.style = t, e.lexical.hasOwnProperty("align") || (e.lexical.align = !0);;) if ((o.length ? o.pop() : Se ? _: x)(n, i)) {
					for (; o.length && o[o.length - 1].lex;) o.pop()();
					return Ee.marked ? Ee.marked: "variable" == n && h(e, i) ? "variable-2": t
				}
			}
			function p() {
				for (var e = arguments.length - 1; e >= 0; e--) Ee.cc.push(arguments[e])
			}
			function m() {
				return p.apply(null, arguments),
				!0
			}
			function g(e) {
				function t(t) {
					for (var n = t; n; n = n.next) if (n.name == e) return ! 0;
					return ! 1
				}
				var n = Ee.state;
				if (Ee.marked = "def", n.context) {
					if (t(n.localVars)) return;
					n.localVars = {
						name: e,
						next: n.localVars
					}
				} else {
					if (t(n.globalVars)) return;
					i.globalVars && (n.globalVars = {
						name: e,
						next: n.globalVars
					})
				}
			}
			function v() {
				Ee.state.context = {
					prev: Ee.state.context,
					vars: Ee.state.localVars
				},
				Ee.state.localVars = $e
			}
			function y() {
				Ee.state.localVars = Ee.state.context.vars,
				Ee.state.context = Ee.state.context.prev
			}
			function b(e, t) {
				var n = function() {
					var n = Ee.state,
					i = n.indented;
					if ("stat" == n.lexical.type) i = n.lexical.indented;
					else for (var r = n.lexical; r && ")" == r.type && r.align; r = r.prev) i = r.indented;
					n.lexical = new d(i, Ee.stream.column(), e, null, n.lexical, t)
				};
				return n.lex = !0,
				n
			}
			function w() {
				var e = Ee.state;
				e.lexical.prev && (")" == e.lexical.type && (e.indented = e.lexical.indented), e.lexical = e.lexical.prev)
			}
			function k(e) {
				function t(n) {
					return n == e ? m() : ";" == e ? p() : m(t)
				}
				return t
			}
			function x(e, t) {
				return "var" == e ? m(b("vardef", t.length), V, k(";"), w) : "keyword a" == e ? m(b("form"), _, x, w) : "keyword b" == e ? m(b("form"), x, w) : "{" == e ? m(b("}"), R, w) : ";" == e ? m() : "if" == e ? ("else" == Ee.state.lexical.info && Ee.state.cc[Ee.state.cc.length - 1] == w && Ee.state.cc.pop()(), m(b("form"), _, x, w, J)) : "function" == e ? m(oe) : "for" == e ? m(b("form"), ee, x, w) : "variable" == e ? m(b("stat"), P) : "switch" == e ? m(b("form"), _, b("}", "switch"), k("{"), R, w, w) : "case" == e ? m(_, k(":")) : "default" == e ? m(k(":")) : "catch" == e ? m(b("form"), v, k("("), ae, k(")"), x, w, y) : "class" == e ? m(b("form"), se, w) : "export" == e ? m(b("stat"), de, w) : "import" == e ? m(b("stat"), he, w) : "module" == e ? m(b("form"), K, b("}"), k("{"), R, w, w) : p(b("stat"), _, k(";"), w)
			}
			function _(e) {
				return S(e, !1)
			}
			function C(e) {
				return S(e, !0)
			}
			function S(e, t) {
				if (Ee.state.fatArrowAt == Ee.stream.start) {
					var n = t ? E: A;
					if ("(" == e) return m(v, b(")"), F(K, ")"), w, k("=>"), n, y);
					if ("variable" == e) return p(v, K, k("=>"), n, y)
				}
				var i = t ? L: D;
				return Ae.hasOwnProperty(e) ? m(i) : "function" == e ? m(oe, i) : "keyword c" == e ? m(t ? T: M) : "(" == e ? m(b(")"), M, ye, k(")"), w, i) : "operator" == e || "spread" == e ? m(t ? C: _) : "[" == e ? m(b("]"), ge, w, i) : "{" == e ? Y(z, "}", null, i) : "quasi" == e ? p(O, i) : "new" == e ? m($(t)) : m()
			}
			function M(e) {
				return e.match(/[;\}\)\],]/) ? p() : p(_)
			}
			function T(e) {
				return e.match(/[;\}\)\],]/) ? p() : p(C)
			}
			function D(e, t) {
				return "," == e ? m(_) : L(e, t, !1)
			}
			function L(e, t, n) {
				var i = 0 == n ? D: L,
				r = 0 == n ? _: C;
				return "=>" == e ? m(v, n ? E: A, y) : "operator" == e ? /\+\+|--/.test(t) ? m(i) : "?" == t ? m(_, k(":"), r) : m(r) : "quasi" == e ? p(O, i) : ";" != e ? "(" == e ? Y(C, ")", "call", i) : "." == e ? m(I, i) : "[" == e ? m(b("]"), M, k("]"), w, i) : void 0 : void 0
			}
			function O(e, t) {
				return "quasi" != e ? p() : "${" != t.slice(t.length - 2) ? m(O) : m(_, N)
			}
			function N(e) {
				return "}" == e ? (Ee.marked = "string-2", Ee.state.tokenize = c, m(O)) : void 0
			}
			function A(e) {
				return u(Ee.stream, Ee.state),
				p("{" == e ? x: _)
			}
			function E(e) {
				return u(Ee.stream, Ee.state),
				p("{" == e ? x: C)
			}
			function $(e) {
				return function(t) {
					return "." == t ? m(e ? j: q) : p(e ? C: _)
				}
			}
			function q(e, t) {
				return "target" == t ? (Ee.marked = "keyword", m(D)) : void 0
			}
			function j(e, t) {
				return "target" == t ? (Ee.marked = "keyword", m(L)) : void 0
			}
			function P(e) {
				return ":" == e ? m(w, x) : p(D, k(";"), w)
			}
			function I(e) {
				return "variable" == e ? (Ee.marked = "property", m()) : void 0
			}
			function z(e, t) {
				return "variable" == e || "keyword" == Ee.style ? (Ee.marked = "property", m("get" == t || "set" == t ? W: H)) : "number" == e || "string" == e ? (Ee.marked = Ce ? "property": Ee.style + " property", m(H)) : "jsonld-keyword" == e ? m(H) : "modifier" == e ? m(z) : "[" == e ? m(_, k("]"), H) : "spread" == e ? m(_) : void 0
			}
			function W(e) {
				return "variable" != e ? p(H) : (Ee.marked = "property", m(oe))
			}
			function H(e) {
				return ":" == e ? m(C) : "(" == e ? p(oe) : void 0
			}
			function F(e, t) {
				function n(i) {
					if ("," == i) {
						var r = Ee.state.lexical;
						return "call" == r.info && (r.pos = (r.pos || 0) + 1),
						m(e, n)
					}
					return i == t ? m() : m(k(t))
				}
				return function(i) {
					return i == t ? m() : p(e, n)
				}
			}
			function Y(e, t, n) {
				for (var i = 3; i < arguments.length; i++) Ee.cc.push(arguments[i]);
				return m(b(t, n), F(e, t), w)
			}
			function R(e) {
				return "}" == e ? m() : p(x, R)
			}
			function U(e) {
				return Me && ":" == e ? m(G) : void 0
			}
			function B(e, t) {
				return "=" == t ? m(C) : void 0
			}
			function G(e) {
				return "variable" == e ? (Ee.marked = "variable-3", m()) : void 0
			}
			function V() {
				return p(K, U, X, Q)
			}
			function K(e, t) {
				return "modifier" == e ? m(K) : "variable" == e ? (g(t), m()) : "spread" == e ? m(K) : "[" == e ? Y(K, "]") : "{" == e ? Y(Z, "}") : void 0
			}
			function Z(e, t) {
				return "variable" != e || Ee.stream.match(/^\s*:/, !1) ? ("variable" == e && (Ee.marked = "property"), "spread" == e ? m(K) : "}" == e ? p() : m(k(":"), K, X)) : (g(t), m(X))
			}
			function X(e, t) {
				return "=" == t ? m(C) : void 0
			}
			function Q(e) {
				return "," == e ? m(V) : void 0
			}
			function J(e, t) {
				return "keyword b" == e && "else" == t ? m(b("form", "else"), x, w) : void 0
			}
			function ee(e) {
				return "(" == e ? m(b(")"), te, k(")"), w) : void 0
			}
			function te(e) {
				return "var" == e ? m(V, k(";"), ie) : ";" == e ? m(ie) : "variable" == e ? m(ne) : p(_, k(";"), ie)
			}
			function ne(e, t) {
				return "in" == t || "of" == t ? (Ee.marked = "keyword", m(_)) : m(D, ie)
			}
			function ie(e, t) {
				return ";" == e ? m(re) : "in" == t || "of" == t ? (Ee.marked = "keyword", m(_)) : p(_, k(";"), re)
			}
			function re(e) {
				")" != e && m(_)
			}
			function oe(e, t) {
				return "*" == t ? (Ee.marked = "keyword", m(oe)) : "variable" == e ? (g(t), m(oe)) : "(" == e ? m(v, b(")"), F(ae, ")"), w, x, y) : void 0
			}
			function ae(e) {
				return "spread" == e ? m(ae) : p(K, U, B)
			}
			function se(e, t) {
				return "variable" == e ? (g(t), m(le)) : void 0
			}
			function le(e, t) {
				return "extends" == t ? m(_, le) : "{" == e ? m(b("}"), ce, w) : void 0
			}
			function ce(e, t) {
				return "variable" == e || "keyword" == Ee.style ? "static" == t ? (Ee.marked = "keyword", m(ce)) : (Ee.marked = "property", "get" == t || "set" == t ? m(ue, oe, ce) : m(oe, ce)) : "*" == t ? (Ee.marked = "keyword", m(ce)) : ";" == e ? m(ce) : "}" == e ? m() : void 0
			}
			function ue(e) {
				return "variable" != e ? p() : (Ee.marked = "property", m())
			}
			function de(e, t) {
				return "*" == t ? (Ee.marked = "keyword", m(me, k(";"))) : "default" == t ? (Ee.marked = "keyword", m(_, k(";"))) : p(x)
			}
			function he(e) {
				return "string" == e ? m() : p(fe, me)
			}
			function fe(e, t) {
				return "{" == e ? Y(fe, "}") : ("variable" == e && g(t), "*" == t && (Ee.marked = "keyword"), m(pe))
			}
			function pe(e, t) {
				return "as" == t ? (Ee.marked = "keyword", m(fe)) : void 0
			}
			function me(e, t) {
				return "from" == t ? (Ee.marked = "keyword", m(_)) : void 0
			}
			function ge(e) {
				return "]" == e ? m() : p(C, ve)
			}
			function ve(e) {
				return "for" == e ? p(ye, k("]")) : "," == e ? m(F(T, "]")) : p(F(C, "]"))
			}
			function ye(e) {
				return "for" == e ? m(ee, ye) : "if" == e ? m(_, ye) : void 0
			}
			function be(e, t) {
				return "operator" == e.lastType || "," == e.lastType || Le.test(t.charAt(0)) || /[,.]/.test(t.charAt(0))
			}
			var we, ke, xe = n.indentUnit,
			_e = i.statementIndent,
			Ce = i.jsonld,
			Se = i.json || Ce,
			Me = i.typescript,
			Te = i.wordCharacters || /[\w$\xa1-\uffff]/,
			De = function() {
				function e(e) {
					return {
						type: e,
						style: "keyword"
					}
				}
				var t = e("keyword a"),
				n = e("keyword b"),
				i = e("keyword c"),
				r = e("operator"),
				o = {
					type: "atom",
					style: "atom"
				},
				a = {
					if: e("if"),
					while: t,
					with: t,
					else: n,
					do: n,
					try: n,
					finally: n,
					return: i,
					break: i,
					continue: i,
					new: e("new"),
					delete: i,
					throw: i,
					debugger: i,
					var: e("var"),
					const: e("var"),
					let: e("var"),
					function: e("function"),
					catch: e("catch"),
					for: e("for"),
					switch: e("switch"),
				case:
					e("case"),
				default:
					e("default"),
					in:r,
					typeof: r,
					instanceof: r,
					true: o,
					false: o,
					null: o,
					undefined: o,
					NaN: o,
					Infinity: o,
					this: e("this"),
					class: e("class"),
					super: e("atom"),
					yield: i,
					export: e("export"),
					import: e("import"),
					extends: i
				};
				if (Me) {
					var s = {
						type: "variable",
						style: "variable-3"
					},
					l = {
						interface: e("class"),
						implements: i,
						namespace: i,
						module: e("module"),
						enum: e("module"),
						public: e("modifier"),
						private: e("modifier"),
						protected: e("modifier"),
						abstract: e("modifier"),
						as: r,
						string: s,
						number: s,
						boolean: s,
						any: s
					};
					for (var c in l) a[c] = l[c]
				}
				return a
			} (),
			Le = /[+\-*&%=<>!?|~^]/,
			Oe = /^@(context|id|value|language|type|container|list|set|reverse|index|base|vocab|graph)"/,
			Ne = "([{}])",
			Ae = {
				atom: !0,
				number: !0,
				variable: !0,
				string: !0,
				regexp: !0,
				this: !0,
				"jsonld-keyword": !0
			},
			Ee = {
				state: null,
				column: null,
				marked: null,
				cc: null
			},
			$e = {
				name: "this",
				next: {
					name: "arguments"
				}
			};
			return w.lex = !0,
			{
				startState: function(e) {
					var t = {
						tokenize: a,
						lastType: "sof",
						cc: [],
						lexical: new d((e || 0) - xe, 0, "block", !1),
						localVars: i.localVars,
						context: i.localVars && {
							vars: i.localVars
						},
						indented: e || 0
					};
					return i.globalVars && "object" == typeof i.globalVars && (t.globalVars = i.globalVars),
					t
				},
				token: function(e, t) {
					if (e.sol() && (t.lexical.hasOwnProperty("align") || (t.lexical.align = !1), t.indented = e.indentation(), u(e, t)), t.tokenize != l && e.eatSpace()) return null;
					var n = t.tokenize(e, t);
					return "comment" == we ? n: (t.lastType = "operator" != we || "++" != ke && "--" != ke ? we: "incdec", f(t, n, we, ke, e))
				},
				indent: function(t, n) {
					if (t.tokenize == l) return e.Pass;
					if (t.tokenize != a) return 0;
					var r = n && n.charAt(0),
					o = t.lexical;
					if (!/^\s*else\b/.test(n)) for (var s = t.cc.length - 1; s >= 0; --s) {
						var c = t.cc[s];
						if (c == w) o = o.prev;
						else if (c != J) break
					}
					"stat" == o.type && "}" == r && (o = o.prev),
					_e && ")" == o.type && "stat" == o.prev.type && (o = o.prev);
					var u = o.type,
					d = r == u;
					return "vardef" == u ? o.indented + ("operator" == t.lastType || "," == t.lastType ? o.info + 1 : 0) : "form" == u && "{" == r ? o.indented: "form" == u ? o.indented + xe: "stat" == u ? o.indented + (be(t, n) ? _e || xe: 0) : "switch" != o.info || d || 0 == i.doubleIndentSwitch ? o.align ? o.column + (d ? 0 : 1) : o.indented + (d ? 0 : xe) : o.indented + (/^(?:case|default)\b/.test(n) ? xe: 2 * xe)
				},
				electricInput: /^\s*(?:case .*?:|default:|\{|\})$/,
				blockCommentStart: Se ? null: "/*",
				blockCommentEnd: Se ? null: "*/",
				lineComment: Se ? null: "//",
				fold: "brace",
				closeBrackets: "()[]{}''\"\"``",
				helperType: Se ? "json": "javascript",
				jsonldMode: Ce,
				jsonMode: Se,
				expressionAllowed: t,
				skipExpression: function(e) {
					var t = e.cc[e.cc.length - 1]; (t == _ || t == C) && e.cc.pop()
				}
			}
		}),
		e.registerHelper("wordChars", "javascript", /[\w$]/),
		e.defineMIME("text/javascript", "javascript"),
		e.defineMIME("text/ecmascript", "javascript"),
		e.defineMIME("application/javascript", "javascript"),
		e.defineMIME("application/x-javascript", "javascript"),
		e.defineMIME("application/ecmascript", "javascript"),
		e.defineMIME("application/json", {
			name: "javascript",
			json: !0
		}),
		e.defineMIME("application/x-json", {
			name: "javascript",
			json: !0
		}),
		e.defineMIME("application/ld+json", {
			name: "javascript",
			jsonld: !0
		}),
		e.defineMIME("text/typescript", {
			name: "javascript",
			typescript: !0
		}),
		e.defineMIME("application/typescript", {
			name: "javascript",
			typescript: !0
		})
	}),
	function(e) {
		"object" == typeof exports && "object" == typeof module ? e(require("../../lib/codemirror"), require("../xml/xml"), require("../meta")) : "function" == typeof define && define.amd ? define(["../../lib/codemirror", "../xml/xml", "../meta"], e) : e(CodeMirror)
	} (function(e) {
		"use strict";
		e.defineMode("markdown",
		function(t, n) {
			function i(n) {
				if (e.findModeByName) {
					var i = e.findModeByName(n);
					i && (n = i.mime || i.mimes[0])
				}
				var r = e.getMode(t, n);
				return "null" == r.name ? null: r
			}
			function r(e, t, n) {
				return t.f = t.inline = n,
				n(e, t)
			}
			function o(e, t, n) {
				return t.f = t.block = n,
				n(e, t)
			}
			function a(e) {
				return ! e || !/\S/.test(e.string)
			}
			function s(e) {
				return e.linkTitle = !1,
				e.em = !1,
				e.strong = !1,
				e.strikethrough = !1,
				e.quote = 0,
				e.indentedCode = !1,
				_ && e.f == c && (e.f = p, e.block = l),
				e.trailingSpace = 0,
				e.trailingSpaceNewLine = !1,
				e.prevLine = e.thisLine,
				e.thisLine = null,
				null
			}
			function l(e, t) {
				var o = e.sol(),
				s = !1 !== t.list,
				l = t.indentedCode;
				t.indentedCode = !1,
				s && (t.indentationDiff >= 0 ? (t.indentationDiff < 4 && (t.indentation -= t.indentationDiff), t.list = null) : t.indentation > 0 ? t.list = null: t.list = !1);
				var c = null;
				if (t.indentationDiff >= 4) return e.skipToEnd(),
				l || a(t.prevLine) ? (t.indentation -= 4, t.indentedCode = !0, C.code) : null;
				if (e.eatSpace()) return null;
				if ((c = e.match(O)) && c[1].length <= 6) return t.header = c[1].length,
				n.highlightFormatting && (t.formatting = "header"),
				t.f = t.inline,
				h(t);
				if (! (a(t.prevLine) || t.quote || s || l) && (c = e.match(N))) return t.header = "=" == c[0].charAt(0) ? 1 : 2,
				n.highlightFormatting && (t.formatting = "header"),
				t.f = t.inline,
				h(t);
				if (e.eat(">")) return t.quote = o ? 1 : t.quote + 1,
				n.highlightFormatting && (t.formatting = "quote"),
				e.eatSpace(),
				h(t);
				if ("[" === e.peek()) return r(e, t, y);
				if (e.match(M, !0)) return t.hr = !0,
				C.hr;
				if ((a(t.prevLine) || s) && (e.match(T, !1) || e.match(D, !1))) {
					var d = null;
					for (e.match(T, !0) ? d = "ul": (e.match(D, !0), d = "ol"), t.indentation = e.column() + e.current().length, t.list = !0; t.listStack && e.column() < t.listStack[t.listStack.length - 1];) t.listStack.pop();
					return t.listStack.push(t.indentation),
					n.taskLists && e.match(L, !1) && (t.taskList = !0),
					t.f = t.inline,
					n.highlightFormatting && (t.formatting = ["list", "list-" + d]),
					h(t)
				}
				return n.fencedCodeBlocks && (c = e.match(E, !0)) ? (t.fencedChars = c[1], t.localMode = i(c[2]), t.localMode && (t.localState = t.localMode.startState()), t.f = t.block = u, n.highlightFormatting && (t.formatting = "code-block"), t.code = -1, h(t)) : r(e, t, t.inline)
			}
			function c(t, n) {
				var i = x.token(t, n.htmlState);
				if (!_) {
					var r = e.innerMode(x, n.htmlState); ("xml" == r.mode.name && null === r.state.tagStart && !r.state.context && r.state.tokenize.isInText || n.md_inside && t.current().indexOf(">") > -1) && (n.f = p, n.block = l, n.htmlState = null)
				}
				return i
			}
			function u(e, t) {
				return t.fencedChars && e.match(t.fencedChars, !1) ? (t.localMode = t.localState = null, t.f = t.block = d, null) : t.localMode ? t.localMode.token(e, t.localState) : (e.skipToEnd(), C.code)
			}
			function d(e, t) {
				e.match(t.fencedChars),
				t.block = l,
				t.f = p,
				t.fencedChars = null,
				n.highlightFormatting && (t.formatting = "code-block"),
				t.code = 1;
				var i = h(t);
				return t.code = 0,
				i
			}
			function h(e) {
				var t = [];
				if (e.formatting) {
					t.push(C.formatting),
					"string" == typeof e.formatting && (e.formatting = [e.formatting]);
					for (var i = 0; i < e.formatting.length; i++) t.push(C.formatting + "-" + e.formatting[i]),
					"header" === e.formatting[i] && t.push(C.formatting + "-" + e.formatting[i] + "-" + e.header),
					"quote" === e.formatting[i] && (!n.maxBlockquoteDepth || n.maxBlockquoteDepth >= e.quote ? t.push(C.formatting + "-" + e.formatting[i] + "-" + e.quote) : t.push("error"))
				}
				if (e.taskOpen) return t.push("meta"),
				t.length ? t.join(" ") : null;
				if (e.taskClosed) return t.push("property"),
				t.length ? t.join(" ") : null;
				if (e.linkHref ? t.push(C.linkHref, "url") : (e.strong && t.push(C.strong), e.em && t.push(C.em), e.strikethrough && t.push(C.strikethrough), e.linkText && t.push(C.linkText), e.code && t.push(C.code)), e.header && t.push(C.header, C.header + "-" + e.header), e.quote && (t.push(C.quote), !n.maxBlockquoteDepth || n.maxBlockquoteDepth >= e.quote ? t.push(C.quote + "-" + e.quote) : t.push(C.quote + "-" + n.maxBlockquoteDepth)), !1 !== e.list) {
					var r = (e.listStack.length - 1) % 3;
					r ? 1 === r ? t.push(C.list2) : t.push(C.list3) : t.push(C.list1)
				}
				return e.trailingSpaceNewLine ? t.push("trailing-space-new-line") : e.trailingSpace && t.push("trailing-space-" + (e.trailingSpace % 2 ? "a": "b")),
				t.length ? t.join(" ") : null
			}
			function f(e, t) {
				return e.match(A, !0) ? h(t) : void 0
			}
			function p(t, i) {
				var r = i.text(t, i);
				if (void 0 !== r) return r;
				if (i.list) return i.list = null,
				h(i);
				if (i.taskList) return "x" !== t.match(L, !0)[1] ? i.taskOpen = !0 : i.taskClosed = !0,
				n.highlightFormatting && (i.formatting = "task"),
				i.taskList = !1,
				h(i);
				if (i.taskOpen = !1, i.taskClosed = !1, i.header && t.match(/^#+$/, !0)) return n.highlightFormatting && (i.formatting = "header"),
				h(i);
				var a = t.sol(),
				s = t.next();
				if (i.linkTitle) {
					i.linkTitle = !1;
					var l = s;
					"(" === s && (l = ")");
					var u = "^\\s*(?:[^" + (l = (l + "").replace(/([.?*+^$[\]\\(){}|-])/g, "\\$1")) + "\\\\]+|\\\\\\\\|\\\\.)" + l;
					if (t.match(new RegExp(u), !0)) return C.linkHref
				}
				if ("`" === s) {
					var d = i.formatting;
					n.highlightFormatting && (i.formatting = "code"),
					t.eatWhile("`");
					var f = t.current().length;
					if (0 == i.code) return i.code = f,
					h(i);
					if (f == i.code) {
						S = h(i);
						return i.code = 0,
						S
					}
					return i.formatting = d,
					h(i)
				}
				if (i.code) return h(i);
				if ("\\" === s && (t.next(), n.highlightFormatting)) {
					var p = h(i),
					v = C.formatting + "-escape";
					return p ? p + " " + v: v
				}
				if ("!" === s && t.match(/\[[^\]]*\] ?(?:\(|\[)/, !1)) return t.match(/\[[^\]]*\]/),
				i.inline = i.f = g,
				C.image;
				if ("[" === s && t.match(/.*\](\(.*\)| ?\[.*\])/, !1)) return i.linkText = !0,
				n.highlightFormatting && (i.formatting = "link"),
				h(i);
				if ("]" === s && i.linkText && t.match(/\(.*\)| ?\[.*\]/, !1)) {
					n.highlightFormatting && (i.formatting = "link");
					p = h(i);
					return i.linkText = !1,
					i.inline = i.f = g,
					p
				}
				if ("<" === s && t.match(/^(https?|ftps?):\/\/(?:[^\\>]|\\.)+>/, !1)) return i.f = i.inline = m,
				n.highlightFormatting && (i.formatting = "link"),
				(p = h(i)) ? p += " ": p = "",
				p + C.linkInline;
				if ("<" === s && t.match(/^[^> \\]+@(?:[^\\>]|\\.)+>/, !1)) return i.f = i.inline = m,
				n.highlightFormatting && (i.formatting = "link"),
				(p = h(i)) ? p += " ": p = "",
				p + C.linkEmail;
				if ("<" === s && t.match(/^(!--|\w)/, !1)) {
					var y = t.string.indexOf(">", t.pos);
					if ( - 1 != y) {
						var b = t.string.substring(t.start, y);
						/markdown\s*=\s*('|"){0,1}1('|"){0,1}/.test(b) && (i.md_inside = !0)
					}
					return t.backUp(1),
					i.htmlState = e.startState(x),
					o(t, i, c)
				}
				if ("<" === s && t.match(/^\/\w*?>/)) return i.md_inside = !1,
				"tag";
				var w = !1;
				if (!n.underscoresBreakWords && "_" === s && "_" !== t.peek() && t.match(/(\w)/, !1)) {
					var k = t.pos - 2;
					if (k >= 0) {
						var _ = t.string.charAt(k);
						"_" !== _ && _.match(/(\w)/, !1) && (w = !0)
					}
				}
				if ("*" === s || "_" === s && !w) if (a && " " === t.peek());
				else {
					if (i.strong === s && t.eat(s)) {
						n.highlightFormatting && (i.formatting = "strong");
						S = h(i);
						return i.strong = !1,
						S
					}
					if (!i.strong && t.eat(s)) return i.strong = s,
					n.highlightFormatting && (i.formatting = "strong"),
					h(i);
					if (i.em === s) {
						n.highlightFormatting && (i.formatting = "em");
						S = h(i);
						return i.em = !1,
						S
					}
					if (!i.em) return i.em = s,
					n.highlightFormatting && (i.formatting = "em"),
					h(i)
				} else if (" " === s && (t.eat("*") || t.eat("_"))) {
					if (" " === t.peek()) return h(i);
					t.backUp(1)
				}
				if (n.strikethrough) if ("~" === s && t.eatWhile(s)) {
					if (i.strikethrough) {
						n.highlightFormatting && (i.formatting = "strikethrough");
						var S = h(i);
						return i.strikethrough = !1,
						S
					}
					if (t.match(/^[^\s]/, !1)) return i.strikethrough = !0,
					n.highlightFormatting && (i.formatting = "strikethrough"),
					h(i)
				} else if (" " === s && t.match(/^~~/, !0)) {
					if (" " === t.peek()) return h(i);
					t.backUp(2)
				}
				return " " === s && (t.match(/ +$/, !1) ? i.trailingSpace++:i.trailingSpace && (i.trailingSpaceNewLine = !0)),
				h(i)
			}
			function m(e, t) {
				if (">" === e.next()) {
					t.f = t.inline = p,
					n.highlightFormatting && (t.formatting = "link");
					var i = h(t);
					return i ? i += " ": i = "",
					i + C.linkInline
				}
				return e.match(/^[^>]+/, !0),
				C.linkInline
			}
			function g(e, t) {
				if (e.eatSpace()) return null;
				var i = e.next();
				return "(" === i || "[" === i ? (t.f = t.inline = v("(" === i ? ")": "]"), n.highlightFormatting && (t.formatting = "link-string"), t.linkHref = !0, h(t)) : "error"
			}
			function v(e) {
				return function(t, i) {
					if (t.next() === e) {
						i.f = i.inline = p,
						n.highlightFormatting && (i.formatting = "link-string");
						var r = h(i);
						return i.linkHref = !1,
						r
					}
					return t.match(k(e), !0) && t.backUp(1),
					i.linkHref = !0,
					h(i)
				}
			}
			function y(e, t) {
				return e.match(/^([^\]\\]|\\.)*\]:/, !1) ? (t.f = b, e.next(), n.highlightFormatting && (t.formatting = "link"), t.linkText = !0, h(t)) : r(e, t, p)
			}
			function b(e, t) {
				if (e.match(/^\]:/, !0)) {
					t.f = t.inline = w,
					n.highlightFormatting && (t.formatting = "link");
					var i = h(t);
					return t.linkText = !1,
					i
				}
				return e.match(/^([^\]\\]|\\.)+/, !0),
				C.linkText
			}
			function w(e, t) {
				return e.eatSpace() ? null: (e.match(/^[^\s]+/, !0), void 0 === e.peek() ? t.linkTitle = !0 : e.match(/^(?:\s+(?:"(?:[^"\\]|\\\\|\\.)+"|'(?:[^'\\]|\\\\|\\.)+'|\((?:[^)\\]|\\\\|\\.)+\)))?/, !0), t.f = t.inline = p, C.linkHref + " url")
			}
			function k(e) {
				return $[e] || (e = (e + "").replace(/([.?*+^$[\]\\(){}|-])/g, "\\$1"), $[e] = new RegExp("^(?:[^\\\\]|\\\\.)*?(" + e + ")")),
				$[e]
			}
			var x = e.getMode(t, "text/html"),
			_ = "null" == x.name;
			void 0 === n.highlightFormatting && (n.highlightFormatting = !1),
			void 0 === n.maxBlockquoteDepth && (n.maxBlockquoteDepth = 0),
			void 0 === n.underscoresBreakWords && (n.underscoresBreakWords = !0),
			void 0 === n.taskLists && (n.taskLists = !1),
			void 0 === n.strikethrough && (n.strikethrough = !1),
			void 0 === n.tokenTypeOverrides && (n.tokenTypeOverrides = {});
			var C = {
				header: "header",
				code: "comment",
				quote: "quote",
				list1: "variable-2",
				list2: "variable-3",
				list3: "keyword",
				hr: "hr",
				image: "tag",
				formatting: "formatting",
				linkInline: "link",
				linkEmail: "link",
				linkText: "link",
				linkHref: "string",
				em: "em",
				strong: "strong",
				strikethrough: "strikethrough"
			};
			for (var S in C) C.hasOwnProperty(S) && n.tokenTypeOverrides[S] && (C[S] = n.tokenTypeOverrides[S]);
			var M = /^([*\-_])(?:\s*\1){2,}\s*$/,
			T = /^[*\-+]\s+/,
			D = /^[0-9]+([.)])\s+/,
			L = /^\[(x| )\](?=\s)/,
			O = n.allowAtxHeaderWithoutSpace ? /^(#+)/: /^(#+)(?: |$)/,
			N = /^ *(?:\={1,}|-{1,})\s*$/,
			A = /^[^#!\[\]*_\\<>` "'(~]+/,
			E = new RegExp("^(" + (!0 === n.fencedCodeBlocks ? "~~~+|```+": n.fencedCodeBlocks) + ")[ \\t]*([\\w+#]*)"),
			$ = [],
			q = {
				startState: function() {
					return {
						f: l,
						prevLine: null,
						thisLine: null,
						block: l,
						htmlState: null,
						indentation: 0,
						inline: p,
						text: f,
						formatting: !1,
						linkText: !1,
						linkHref: !1,
						linkTitle: !1,
						code: 0,
						em: !1,
						strong: !1,
						header: 0,
						hr: !1,
						taskList: !1,
						list: !1,
						listStack: [],
						quote: 0,
						trailingSpace: 0,
						trailingSpaceNewLine: !1,
						strikethrough: !1,
						fencedChars: null
					}
				},
				copyState: function(t) {
					return {
						f: t.f,
						prevLine: t.prevLine,
						thisLine: t.thisLine,
						block: t.block,
						htmlState: t.htmlState && e.copyState(x, t.htmlState),
						indentation: t.indentation,
						localMode: t.localMode,
						localState: t.localMode ? e.copyState(t.localMode, t.localState) : null,
						inline: t.inline,
						text: t.text,
						formatting: !1,
						linkTitle: t.linkTitle,
						code: t.code,
						em: t.em,
						strong: t.strong,
						strikethrough: t.strikethrough,
						header: t.header,
						hr: t.hr,
						taskList: t.taskList,
						list: t.list,
						listStack: t.listStack.slice(0),
						quote: t.quote,
						indentedCode: t.indentedCode,
						trailingSpace: t.trailingSpace,
						trailingSpaceNewLine: t.trailingSpaceNewLine,
						md_inside: t.md_inside,
						fencedChars: t.fencedChars
					}
				},
				token: function(e, t) {
					if (t.formatting = !1, e != t.thisLine) {
						var n = t.header || t.hr;
						if (t.header = 0, t.hr = !1, e.match(/^\s*$/, !0) || n) {
							if (s(t), !n) return null;
							t.prevLine = null
						}
						t.prevLine = t.thisLine,
						t.thisLine = e,
						t.taskList = !1,
						t.trailingSpace = 0,
						t.trailingSpaceNewLine = !1,
						t.f = t.block;
						var i = e.match(/^\s*/, !0)[0].replace(/\t/g, "    ").length;
						if (t.indentationDiff = Math.min(i - t.indentation, 4), t.indentation = t.indentation + t.indentationDiff, i > 0) return null
					}
					return t.f(e, t)
				},
				innerMode: function(e) {
					return e.block == c ? {
						state: e.htmlState,
						mode: x
					}: e.localState ? {
						state: e.localState,
						mode: e.localMode
					}: {
						state: e,
						mode: q
					}
				},
				blankLine: s,
				getType: h,
				fold: "markdown"
			};
			return q
		},
		"xml"),
		e.defineMIME("text/x-markdown", "markdown")
	}),
	function(e) {
		"object" == typeof exports && "object" == typeof module ? e(require("../../lib/codemirror"), require("../htmlmixed/htmlmixed"), require("../clike/clike")) : "function" == typeof define && define.amd ? define(["../../lib/codemirror", "../htmlmixed/htmlmixed", "../clike/clike"], e) : e(CodeMirror)
	} (function(e) {
		"use strict";
		function t(e) {
			for (var t = {},
			n = e.split(" "), i = 0; i < n.length; ++i) t[n[i]] = !0;
			return t
		}
		function n(e, t, r) {
			return 0 == e.length ? i(t) : function(o, a) {
				for (var s = e[0], l = 0; l < s.length; l++) if (o.match(s[l][0])) return a.tokenize = n(e.slice(1), t),
				s[l][1];
				return a.tokenize = i(t, r),
				"string"
			}
		}
		function i(e, t) {
			return function(n, i) {
				return r(n, i, e, t)
			}
		}
		function r(e, t, i, r) {
			if (!1 !== r && e.match("${", !1) || e.match("{$", !1)) return t.tokenize = null,
			"string";
			if (!1 !== r && e.match(/^\$[a-zA-Z_][a-zA-Z0-9_]*/)) return e.match("[", !1) && (t.tokenize = n([[["[", null]], [[/\d[\w\.]*/, "number"], [/\$[a-zA-Z_][a-zA-Z0-9_]*/, "variable-2"], [/[\w\$]+/, "variable"]], [["]", null]]], i, r)),
			e.match(/\-\>\w/, !1) && (t.tokenize = n([[["->", null]], [[/[\w]+/, "variable"]]], i, r)),
			"variable-2";
			for (var o = !1; ! e.eol() && (o || !1 === r || !e.match("{$", !1) && !e.match(/^(\$[a-zA-Z_][a-zA-Z0-9_]*|\$\{)/, !1));) {
				if (!o && e.match(i)) {
					t.tokenize = null,
					t.tokStack.pop(),
					t.tokStack.pop();
					break
				}
				o = "\\" == e.next() && !o
			}
			return "string"
		}
		var o = "abstract and array as break case catch class clone const continue declare default do else elseif enddeclare endfor endforeach endif endswitch endwhile extends final for foreach function global goto if implements interface instanceof namespace new or private protected public static switch throw trait try use var while xor die echo empty exit eval include include_once isset list require require_once return print unset __halt_compiler self static parent yield insteadof finally",
		a = "true false null TRUE FALSE NULL __CLASS__ __DIR__ __FILE__ __LINE__ __METHOD__ __FUNCTION__ __NAMESPACE__ __TRAIT__",
		s = "func_num_args func_get_arg func_get_args strlen strcmp strncmp strcasecmp strncasecmp each error_reporting define defined trigger_error user_error set_error_handler restore_error_handler get_declared_classes get_loaded_extensions extension_loaded get_extension_funcs debug_backtrace constant bin2hex hex2bin sleep usleep time mktime gmmktime strftime gmstrftime strtotime date gmdate getdate localtime checkdate flush wordwrap htmlspecialchars htmlentities html_entity_decode md5 md5_file crc32 getimagesize image_type_to_mime_type phpinfo phpversion phpcredits strnatcmp strnatcasecmp substr_count strspn strcspn strtok strtoupper strtolower strpos strrpos strrev hebrev hebrevc nl2br basename dirname pathinfo stripslashes stripcslashes strstr stristr strrchr str_shuffle str_word_count strcoll substr substr_replace quotemeta ucfirst ucwords strtr addslashes addcslashes rtrim str_replace str_repeat count_chars chunk_split trim ltrim strip_tags similar_text explode implode setlocale localeconv parse_str str_pad chop strchr sprintf printf vprintf vsprintf sscanf fscanf parse_url urlencode urldecode rawurlencode rawurldecode readlink linkinfo link unlink exec system escapeshellcmd escapeshellarg passthru shell_exec proc_open proc_close rand srand getrandmax mt_rand mt_srand mt_getrandmax base64_decode base64_encode abs ceil floor round is_finite is_nan is_infinite bindec hexdec octdec decbin decoct dechex base_convert number_format fmod ip2long long2ip getenv putenv getopt microtime gettimeofday getrusage uniqid quoted_printable_decode set_time_limit get_cfg_var magic_quotes_runtime set_magic_quotes_runtime get_magic_quotes_gpc get_magic_quotes_runtime import_request_variables error_log serialize unserialize memory_get_usage var_dump var_export debug_zval_dump print_r highlight_file show_source highlight_string ini_get ini_get_all ini_set ini_alter ini_restore get_include_path set_include_path restore_include_path setcookie header headers_sent connection_aborted connection_status ignore_user_abort parse_ini_file is_uploaded_file move_uploaded_file intval floatval doubleval strval gettype settype is_null is_resource is_bool is_long is_float is_int is_integer is_double is_real is_numeric is_string is_array is_object is_scalar ereg ereg_replace eregi eregi_replace split spliti join sql_regcase dl pclose popen readfile rewind rmdir umask fclose feof fgetc fgets fgetss fread fopen fpassthru ftruncate fstat fseek ftell fflush fwrite fputs mkdir rename copy tempnam tmpfile file file_get_contents file_put_contents stream_select stream_context_create stream_context_set_params stream_context_set_option stream_context_get_options stream_filter_prepend stream_filter_append fgetcsv flock get_meta_tags stream_set_write_buffer set_file_buffer set_socket_blocking stream_set_blocking socket_set_blocking stream_get_meta_data stream_register_wrapper stream_wrapper_register stream_set_timeout socket_set_timeout socket_get_status realpath fnmatch fsockopen pfsockopen pack unpack get_browser crypt opendir closedir chdir getcwd rewinddir readdir dir glob fileatime filectime filegroup fileinode filemtime fileowner fileperms filesize filetype file_exists is_writable is_writeable is_readable is_executable is_file is_dir is_link stat lstat chown touch clearstatcache mail ob_start ob_flush ob_clean ob_end_flush ob_end_clean ob_get_flush ob_get_clean ob_get_length ob_get_level ob_get_status ob_get_contents ob_implicit_flush ob_list_handlers ksort krsort natsort natcasesort asort arsort sort rsort usort uasort uksort shuffle array_walk count end prev next reset current key min max in_array array_search extract compact array_fill range array_multisort array_push array_pop array_shift array_unshift array_splice array_slice array_merge array_merge_recursive array_keys array_values array_count_values array_reverse array_reduce array_pad array_flip array_change_key_case array_rand array_unique array_intersect array_intersect_assoc array_diff array_diff_assoc array_sum array_filter array_map array_chunk array_key_exists pos sizeof key_exists assert assert_options version_compare ftok str_rot13 aggregate session_name session_module_name session_save_path session_id session_regenerate_id session_decode session_register session_unregister session_is_registered session_encode session_start session_destroy session_unset session_set_save_handler session_cache_limiter session_cache_expire session_set_cookie_params session_get_cookie_params session_write_close preg_match preg_match_all preg_replace preg_replace_callback preg_split preg_quote preg_grep overload ctype_alnum ctype_alpha ctype_cntrl ctype_digit ctype_lower ctype_graph ctype_print ctype_punct ctype_space ctype_upper ctype_xdigit virtual apache_request_headers apache_note apache_lookup_uri apache_child_terminate apache_setenv apache_response_headers apache_get_version getallheaders mysql_connect mysql_pconnect mysql_close mysql_select_db mysql_create_db mysql_drop_db mysql_query mysql_unbuffered_query mysql_db_query mysql_list_dbs mysql_list_tables mysql_list_fields mysql_list_processes mysql_error mysql_errno mysql_affected_rows mysql_insert_id mysql_result mysql_num_rows mysql_num_fields mysql_fetch_row mysql_fetch_array mysql_fetch_assoc mysql_fetch_object mysql_data_seek mysql_fetch_lengths mysql_fetch_field mysql_field_seek mysql_free_result mysql_field_name mysql_field_table mysql_field_len mysql_field_type mysql_field_flags mysql_escape_string mysql_real_escape_string mysql_stat mysql_thread_id mysql_client_encoding mysql_get_client_info mysql_get_host_info mysql_get_proto_info mysql_get_server_info mysql_info mysql mysql_fieldname mysql_fieldtable mysql_fieldlen mysql_fieldtype mysql_fieldflags mysql_selectdb mysql_createdb mysql_dropdb mysql_freeresult mysql_numfields mysql_numrows mysql_listdbs mysql_listtables mysql_listfields mysql_db_name mysql_dbname mysql_tablename mysql_table_name pg_connect pg_pconnect pg_close pg_connection_status pg_connection_busy pg_connection_reset pg_host pg_dbname pg_port pg_tty pg_options pg_ping pg_query pg_send_query pg_cancel_query pg_fetch_result pg_fetch_row pg_fetch_assoc pg_fetch_array pg_fetch_object pg_fetch_all pg_affected_rows pg_get_result pg_result_seek pg_result_status pg_free_result pg_last_oid pg_num_rows pg_num_fields pg_field_name pg_field_num pg_field_size pg_field_type pg_field_prtlen pg_field_is_null pg_get_notify pg_get_pid pg_result_error pg_last_error pg_last_notice pg_put_line pg_end_copy pg_copy_to pg_copy_from pg_trace pg_untrace pg_lo_create pg_lo_unlink pg_lo_open pg_lo_close pg_lo_read pg_lo_write pg_lo_read_all pg_lo_import pg_lo_export pg_lo_seek pg_lo_tell pg_escape_string pg_escape_bytea pg_unescape_bytea pg_client_encoding pg_set_client_encoding pg_meta_data pg_convert pg_insert pg_update pg_delete pg_select pg_exec pg_getlastoid pg_cmdtuples pg_errormessage pg_numrows pg_numfields pg_fieldname pg_fieldsize pg_fieldtype pg_fieldnum pg_fieldprtlen pg_fieldisnull pg_freeresult pg_result pg_loreadall pg_locreate pg_lounlink pg_loopen pg_loclose pg_loread pg_lowrite pg_loimport pg_loexport http_response_code get_declared_traits getimagesizefromstring socket_import_stream stream_set_chunk_size trait_exists header_register_callback class_uses session_status session_register_shutdown echo print global static exit array empty eval isset unset die include require include_once require_once json_decode json_encode json_last_error json_last_error_msg curl_close curl_copy_handle curl_errno curl_error curl_escape curl_exec curl_file_create curl_getinfo curl_init curl_multi_add_handle curl_multi_close curl_multi_exec curl_multi_getcontent curl_multi_info_read curl_multi_init curl_multi_remove_handle curl_multi_select curl_multi_setopt curl_multi_strerror curl_pause curl_reset curl_setopt_array curl_setopt curl_share_close curl_share_init curl_share_setopt curl_strerror curl_unescape curl_version mysqli_affected_rows mysqli_autocommit mysqli_change_user mysqli_character_set_name mysqli_close mysqli_commit mysqli_connect_errno mysqli_connect_error mysqli_connect mysqli_data_seek mysqli_debug mysqli_dump_debug_info mysqli_errno mysqli_error_list mysqli_error mysqli_fetch_all mysqli_fetch_array mysqli_fetch_assoc mysqli_fetch_field_direct mysqli_fetch_field mysqli_fetch_fields mysqli_fetch_lengths mysqli_fetch_object mysqli_fetch_row mysqli_field_count mysqli_field_seek mysqli_field_tell mysqli_free_result mysqli_get_charset mysqli_get_client_info mysqli_get_client_stats mysqli_get_client_version mysqli_get_connection_stats mysqli_get_host_info mysqli_get_proto_info mysqli_get_server_info mysqli_get_server_version mysqli_info mysqli_init mysqli_insert_id mysqli_kill mysqli_more_results mysqli_multi_query mysqli_next_result mysqli_num_fields mysqli_num_rows mysqli_options mysqli_ping mysqli_prepare mysqli_query mysqli_real_connect mysqli_real_escape_string mysqli_real_query mysqli_reap_async_query mysqli_refresh mysqli_rollback mysqli_select_db mysqli_set_charset mysqli_set_local_infile_default mysqli_set_local_infile_handler mysqli_sqlstate mysqli_ssl_set mysqli_stat mysqli_stmt_init mysqli_store_result mysqli_thread_id mysqli_thread_safe mysqli_use_result mysqli_warning_count";
		e.registerHelper("hintWords", "php", [o, a, s].join(" ").split(" ")),
		e.registerHelper("wordChars", "php", /[\w$]/);
		var l = {
			name: "clike",
			helperType: "php",
			keywords: t(o),
			blockKeywords: t("catch do else elseif for foreach if switch try while finally"),
			defKeywords: t("class function interface namespace trait"),
			atoms: t(a),
			builtin: t(s),
			multiLineStrings: !0,
			hooks: {
				$: function(e) {
					return e.eatWhile(/[\w\$_]/),
					"variable-2"
				},
				"<": function(e, t) {
					var n;
					if (n = e.match(/<<\s*/)) {
						var r = e.eat(/['"]/);
						e.eatWhile(/[\w\.]/);
						var o = e.current().slice(n[0].length + (r ? 2 : 1));
						if (r && e.eat(r), o) return (t.tokStack || (t.tokStack = [])).push(o, 0),
						t.tokenize = i(o, "'" != r),
						"string"
					}
					return ! 1
				},
				"#": function(e) {
					for (; ! e.eol() && !e.match("?>", !1);) e.next();
					return "comment"
				},
				"/": function(e) {
					if (e.eat("/")) {
						for (; ! e.eol() && !e.match("?>", !1);) e.next();
						return "comment"
					}
					return ! 1
				},
				'"': function(e, t) {
					return (t.tokStack || (t.tokStack = [])).push('"', 0),
					t.tokenize = i('"'),
					"string"
				},
				"{": function(e, t) {
					return t.tokStack && t.tokStack.length && t.tokStack[t.tokStack.length - 1]++,
					!1
				},
				"}": function(e, t) {
					return t.tokStack && t.tokStack.length > 0 && !--t.tokStack[t.tokStack.length - 1] && (t.tokenize = i(t.tokStack[t.tokStack.length - 2])),
					!1
				}
			}
		};
		e.defineMode("php",
		function(t, n) {
			var i = e.getMode(t, "text/html"),
			r = e.getMode(t, l);
			return {
				startState: function() {
					var t = e.startState(i),
					o = n.startOpen ? e.startState(r) : null;
					return {
						html: t,
						php: o,
						curMode: n.startOpen ? r: i,
						curState: n.startOpen ? o: t,
						pending: null
					}
				},
				copyState: function(t) {
					var n, o = t.html,
					a = e.copyState(i, o),
					s = t.php,
					l = s && e.copyState(r, s);
					return n = t.curMode == i ? a: l,
					{
						html: a,
						php: l,
						curMode: t.curMode,
						curState: n,
						pending: t.pending
					}
				},
				token: function(t, n) {
					var o = n.curMode == r;
					if (t.sol() && n.pending && '"' != n.pending && "'" != n.pending && (n.pending = null), o) return o && null == n.php.tokenize && t.match("?>") ? (n.curMode = i, n.curState = n.html, n.php.context.prev || (n.php = null), "meta") : r.token(t, n.curState);
					if (t.match(/^<\?\w*/)) return n.curMode = r,
					n.php || (n.php = e.startState(r, i.indent(n.html, ""))),
					n.curState = n.php,
					"meta";
					if ('"' == n.pending || "'" == n.pending) {
						for (; ! t.eol() && t.next() != n.pending;);
						a = "string"
					} else if (n.pending && t.pos < n.pending.end) t.pos = n.pending.end,
					a = n.pending.style;
					else var a = i.token(t, n.curState);
					n.pending && (n.pending = null);
					var s, l = t.current(),
					c = l.search(/<\?/);
					return - 1 != c && ("string" == a && (s = l.match(/[\'\"]$/)) && !/\?>/.test(l) ? n.pending = s[0] : n.pending = {
						end: t.pos,
						style: a
					},
					t.backUp(l.length - c)),
					a
				},
				indent: function(e, t) {
					return e.curMode != r && /^\s*<\//.test(t) || e.curMode == r && /^\?>/.test(t) ? i.indent(e.html, t) : e.curMode.indent(e.curState, t)
				},
				blockCommentStart: "/*",
				blockCommentEnd: "*/",
				lineComment: "//",
				innerMode: function(e) {
					return {
						state: e.curState,
						mode: e.curMode
					}
				}
			}
		},
		"htmlmixed", "clike"),
		e.defineMIME("application/x-httpd-php", "php"),
		e.defineMIME("application/x-httpd-php-open", {
			name: "php",
			startOpen: !0
		}),
		e.defineMIME("text/x-php", l)
	}),
	function(e) {
		"object" == typeof exports && "object" == typeof module ? e(require("../../lib/codemirror")) : "function" == typeof define && define.amd ? define(["../../lib/codemirror"], e) : e(CodeMirror)
	} (function(e) {
		"use strict";
		e.defineMode("sql",
		function(t, n) {
			function i(e, t) {
				var n = e.next();
				if (p[n]) {
					var i = p[n](e, t);
					if (!1 !== i) return i
				}
				if (1 == f.hexNumber && ("0" == n && e.match(/^[xX][0-9a-fA-F]+/) || ("x" == n || "X" == n) && e.match(/^'[0-9a-fA-F]+'/))) return "number";
				if (1 == f.binaryNumber && (("b" == n || "B" == n) && e.match(/^'[01]+'/) || "0" == n && e.match(/^b[01]+/))) return "number";
				if (n.charCodeAt(0) > 47 && n.charCodeAt(0) < 58) return e.match(/^[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?/),
				1 == f.decimallessFloat && e.eat("."),
				"number";
				if ("?" == n && (e.eatSpace() || e.eol() || e.eat(";"))) return "variable-3";
				if ("'" == n || '"' == n && f.doubleQuote) return t.tokenize = r(n),
				t.tokenize(e, t);
				if ((1 == f.nCharCast && ("n" == n || "N" == n) || 1 == f.charsetCast && "_" == n && e.match(/[a-z][a-z0-9]*/i)) && ("'" == e.peek() || '"' == e.peek())) return "keyword";
				if (/^[\(\),\;\[\]]/.test(n)) return null;
				if (f.commentSlashSlash && "/" == n && e.eat("/")) return e.skipToEnd(),
				"comment";
				if (f.commentHash && "#" == n || "-" == n && e.eat("-") && (!f.commentSpaceRequired || e.eat(" "))) return e.skipToEnd(),
				"comment";
				if ("/" == n && e.eat("*")) return t.tokenize = o,
				t.tokenize(e, t);
				if ("." != n) {
					if (h.test(n)) return e.eatWhile(h),
					null;
					if ("{" == n && (e.match(/^( )*(d|D|t|T|ts|TS)( )*'[^']*'( )*}/) || e.match(/^( )*(d|D|t|T|ts|TS)( )*"[^"]*"( )*}/))) return "number";
					e.eatWhile(/^[_\w\d]/);
					var a = e.current().toLowerCase();
					return m.hasOwnProperty(a) && (e.match(/^( )+'[^']*'/) || e.match(/^( )+"[^"]*"/)) ? "number": c.hasOwnProperty(a) ? "atom": u.hasOwnProperty(a) ? "builtin": d.hasOwnProperty(a) ? "keyword": l.hasOwnProperty(a) ? "string-2": null
				}
				return 1 == f.zerolessFloat && e.match(/^(?:\d+(?:e[+-]?\d+)?)/i) ? "number": 1 == f.ODBCdotTable && e.match(/^[a-zA-Z_]+/) ? "variable-2": void 0
			}
			function r(e) {
				return function(t, n) {
					for (var r, o = !1; null != (r = t.next());) {
						if (r == e && !o) {
							n.tokenize = i;
							break
						}
						o = !o && "\\" == r
					}
					return "string"
				}
			}
			function o(e, t) {
				for (;;) {
					if (!e.skipTo("*")) {
						e.skipToEnd();
						break
					}
					if (e.next(), e.eat("/")) {
						t.tokenize = i;
						break
					}
				}
				return "comment"
			}
			function a(e, t, n) {
				t.context = {
					prev: t.context,
					indent: e.indentation(),
					col: e.column(),
					type: n
				}
			}
			function s(e) {
				e.indent = e.context.indent,
				e.context = e.context.prev
			}
			var l = n.client || {},
			c = n.atoms || {
				false: !0,
				true: !0,
				null: !0
			},
			u = n.builtin || {},
			d = n.keywords || {},
			h = n.operatorChars || /^[*+\-%<>!=&|~^]/,
			f = n.support || {},
			p = n.hooks || {},
			m = n.dateSQL || {
				date: !0,
				time: !0,
				timestamp: !0
			};
			return {
				startState: function() {
					return {
						tokenize: i,
						context: null
					}
				},
				token: function(e, t) {
					if (e.sol() && t.context && null == t.context.align && (t.context.align = !1), e.eatSpace()) return null;
					var n = t.tokenize(e, t);
					if ("comment" == n) return n;
					t.context && null == t.context.align && (t.context.align = !0);
					var i = e.current();
					return "(" == i ? a(e, t, ")") : "[" == i ? a(e, t, "]") : t.context && t.context.type == i && s(t),
					n
				},
				indent: function(n, i) {
					var r = n.context;
					if (!r) return e.Pass;
					var o = i.charAt(0) == r.type;
					return r.align ? r.col + (o ? 0 : 1) : r.indent + (o ? 0 : t.indentUnit)
				},
				blockCommentStart: "/*",
				blockCommentEnd: "*/",
				lineComment: f.commentSlashSlash ? "//": f.commentHash ? "#": null
			}
		}),
		function() {
			function t(e) {
				for (var t; null != (t = e.next());) if ("`" == t && !e.eat("`")) return "variable-2";
				return e.backUp(e.current().length - 1),
				e.eatWhile(/\w/) ? "variable-2": null
			}
			function n(e) {
				return e.eat("@") && (e.match(/^session\./), e.match(/^local\./), e.match(/^global\./)),
				e.eat("'") ? (e.match(/^.*'/), "variable-2") : e.eat('"') ? (e.match(/^.*"/), "variable-2") : e.eat("`") ? (e.match(/^.*`/), "variable-2") : e.match(/^[0-9a-zA-Z$\.\_]+/) ? "variable-2": null
			}
			function i(e) {
				return e.eat("N") ? "atom": e.match(/^[a-zA-Z.#!?]/) ? "variable-2": null
			}
			function r(e) {
				for (var t = {},
				n = e.split(" "), i = 0; i < n.length; ++i) t[n[i]] = !0;
				return t
			}
			var o = "alter and as asc between by count create delete desc distinct drop from group having in insert into is join like not on or order select set table union update values where limit ";
			e.defineMIME("text/x-sql", {
				name: "sql",
				keywords: r(o + "begin"),
				builtin: r("bool boolean bit blob enum long longblob longtext medium mediumblob mediumint mediumtext time timestamp tinyblob tinyint tinytext text bigint int int1 int2 int3 int4 int8 integer float float4 float8 double char varbinary varchar varcharacter precision real date datetime year unsigned signed decimal numeric"),
				atoms: r("false true null unknown"),
				operatorChars: /^[*+\-%<>!=]/,
				dateSQL: r("date time timestamp"),
				support: r("ODBCdotTable doubleQuote binaryNumber hexNumber")
			}),
			e.defineMIME("text/x-mssql", {
				name: "sql",
				client: r("charset clear connect edit ego exit go help nopager notee nowarning pager print prompt quit rehash source status system tee"),
				keywords: r(o + "begin trigger proc view index for add constraint key primary foreign collate clustered nonclustered declare"),
				builtin: r("bigint numeric bit smallint decimal smallmoney int tinyint money float real char varchar text nchar nvarchar ntext binary varbinary image cursor timestamp hierarchyid uniqueidentifier sql_variant xml table "),
				atoms: r("false true null unknown"),
				operatorChars: /^[*+\-%<>!=]/,
				dateSQL: r("date datetimeoffset datetime2 smalldatetime datetime time"),
				hooks: {
					"@": n
				}
			}),
			e.defineMIME("text/x-mysql", {
				name: "sql",
				client: r("charset clear connect edit ego exit go help nopager notee nowarning pager print prompt quit rehash source status system tee"),
				keywords: r(o + "accessible action add after algorithm all analyze asensitive at authors auto_increment autocommit avg avg_row_length before binary binlog both btree cache call cascade cascaded case catalog_name chain change changed character check checkpoint checksum class_origin client_statistics close coalesce code collate collation collations column columns comment commit committed completion concurrent condition connection consistent constraint contains continue contributors convert cross current current_date current_time current_timestamp current_user cursor data database databases day_hour day_microsecond day_minute day_second deallocate dec declare default delay_key_write delayed delimiter des_key_file describe deterministic dev_pop dev_samp deviance diagnostics directory disable discard distinctrow div dual dumpfile each elseif enable enclosed end ends engine engines enum errors escape escaped even event events every execute exists exit explain extended fast fetch field fields first flush for force foreign found_rows full fulltext function general get global grant grants group group_concat handler hash help high_priority hosts hour_microsecond hour_minute hour_second if ignore ignore_server_ids import index index_statistics infile inner innodb inout insensitive insert_method install interval invoker isolation iterate key keys kill language last leading leave left level limit linear lines list load local localtime localtimestamp lock logs low_priority master master_heartbeat_period master_ssl_verify_server_cert masters match max max_rows maxvalue message_text middleint migrate min min_rows minute_microsecond minute_second mod mode modifies modify mutex mysql_errno natural next no no_write_to_binlog offline offset one online open optimize option optionally out outer outfile pack_keys parser partition partitions password phase plugin plugins prepare preserve prev primary privileges procedure processlist profile profiles purge query quick range read read_write reads real rebuild recover references regexp relaylog release remove rename reorganize repair repeatable replace require resignal restrict resume return returns revoke right rlike rollback rollup row row_format rtree savepoint schedule schema schema_name schemas second_microsecond security sensitive separator serializable server session share show signal slave slow smallint snapshot soname spatial specific sql sql_big_result sql_buffer_result sql_cache sql_calc_found_rows sql_no_cache sql_small_result sqlexception sqlstate sqlwarning ssl start starting starts status std stddev stddev_pop stddev_samp storage straight_join subclass_origin sum suspend table_name table_statistics tables tablespace temporary terminated to trailing transaction trigger triggers truncate uncommitted undo uninstall unique unlock upgrade usage use use_frm user user_resources user_statistics using utc_date utc_time utc_timestamp value variables varying view views warnings when while with work write xa xor year_month zerofill begin do then else loop repeat"),
				builtin: r("bool boolean bit blob decimal double float long longblob longtext medium mediumblob mediumint mediumtext time timestamp tinyblob tinyint tinytext text bigint int int1 int2 int3 int4 int8 integer float float4 float8 double char varbinary varchar varcharacter precision date datetime year unsigned signed numeric"),
				atoms: r("false true null unknown"),
				operatorChars: /^[*+\-%<>!=&|^]/,
				dateSQL: r("date time timestamp"),
				support: r("ODBCdotTable decimallessFloat zerolessFloat binaryNumber hexNumber doubleQuote nCharCast charsetCast commentHash commentSpaceRequired"),
				hooks: {
					"@": n,
					"`": t,
					"\\": i
				}
			}),
			e.defineMIME("text/x-mariadb", {
				name: "sql",
				client: r("charset clear connect edit ego exit go help nopager notee nowarning pager print prompt quit rehash source status system tee"),
				keywords: r(o + "accessible action add after algorithm all always analyze asensitive at authors auto_increment autocommit avg avg_row_length before binary binlog both btree cache call cascade cascaded case catalog_name chain change changed character check checkpoint checksum class_origin client_statistics close coalesce code collate collation collations column columns comment commit committed completion concurrent condition connection consistent constraint contains continue contributors convert cross current current_date current_time current_timestamp current_user cursor data database databases day_hour day_microsecond day_minute day_second deallocate dec declare default delay_key_write delayed delimiter des_key_file describe deterministic dev_pop dev_samp deviance diagnostics directory disable discard distinctrow div dual dumpfile each elseif enable enclosed end ends engine engines enum errors escape escaped even event events every execute exists exit explain extended fast fetch field fields first flush for force foreign found_rows full fulltext function general generated get global grant grants group groupby_concat handler hard hash help high_priority hosts hour_microsecond hour_minute hour_second if ignore ignore_server_ids import index index_statistics infile inner innodb inout insensitive insert_method install interval invoker isolation iterate key keys kill language last leading leave left level limit linear lines list load local localtime localtimestamp lock logs low_priority master master_heartbeat_period master_ssl_verify_server_cert masters match max max_rows maxvalue message_text middleint migrate min min_rows minute_microsecond minute_second mod mode modifies modify mutex mysql_errno natural next no no_write_to_binlog offline offset one online open optimize option optionally out outer outfile pack_keys parser partition partitions password persistent phase plugin plugins prepare preserve prev primary privileges procedure processlist profile profiles purge query quick range read read_write reads real rebuild recover references regexp relaylog release remove rename reorganize repair repeatable replace require resignal restrict resume return returns revoke right rlike rollback rollup row row_format rtree savepoint schedule schema schema_name schemas second_microsecond security sensitive separator serializable server session share show shutdown signal slave slow smallint snapshot soft soname spatial specific sql sql_big_result sql_buffer_result sql_cache sql_calc_found_rows sql_no_cache sql_small_result sqlexception sqlstate sqlwarning ssl start starting starts status std stddev stddev_pop stddev_samp storage straight_join subclass_origin sum suspend table_name table_statistics tables tablespace temporary terminated to trailing transaction trigger triggers truncate uncommitted undo uninstall unique unlock upgrade usage use use_frm user user_resources user_statistics using utc_date utc_time utc_timestamp value variables varying view views virtual warnings when while with work write xa xor year_month zerofill begin do then else loop repeat"),
				builtin: r("bool boolean bit blob decimal double float long longblob longtext medium mediumblob mediumint mediumtext time timestamp tinyblob tinyint tinytext text bigint int int1 int2 int3 int4 int8 integer float float4 float8 double char varbinary varchar varcharacter precision date datetime year unsigned signed numeric"),
				atoms: r("false true null unknown"),
				operatorChars: /^[*+\-%<>!=&|^]/,
				dateSQL: r("date time timestamp"),
				support: r("ODBCdotTable decimallessFloat zerolessFloat binaryNumber hexNumber doubleQuote nCharCast charsetCast commentHash commentSpaceRequired"),
				hooks: {
					"@": n,
					"`": t,
					"\\": i
				}
			}),
			e.defineMIME("text/x-cassandra", {
				name: "sql",
				client: {},
				keywords: r("add all allow alter and any apply as asc authorize batch begin by clustering columnfamily compact consistency count create custom delete desc distinct drop each_quorum exists filtering from grant if in index insert into key keyspace keyspaces level limit local_one local_quorum modify nan norecursive nosuperuser not of on one order password permission permissions primary quorum rename revoke schema select set storage superuser table three to token truncate ttl two type unlogged update use user users using values where with writetime"),
				builtin: r("ascii bigint blob boolean counter decimal double float frozen inet int list map static text timestamp timeuuid tuple uuid varchar varint"),
				atoms: r("false true infinity NaN"),
				operatorChars: /^[<>=]/,
				dateSQL: {},
				support: r("commentSlashSlash decimallessFloat"),
				hooks: {}
			}),
			e.defineMIME("text/x-plsql", {
				name: "sql",
				client: r("appinfo arraysize autocommit autoprint autorecovery autotrace blockterminator break btitle cmdsep colsep compatibility compute concat copycommit copytypecheck define describe echo editfile embedded escape exec execute feedback flagger flush heading headsep instance linesize lno loboffset logsource long longchunksize markup native newpage numformat numwidth pagesize pause pno recsep recsepchar release repfooter repheader serveroutput shiftinout show showmode size spool sqlblanklines sqlcase sqlcode sqlcontinue sqlnumber sqlpluscompatibility sqlprefix sqlprompt sqlterminator suffix tab term termout time timing trimout trimspool ttitle underline verify version wrap"),
				keywords: r("abort accept access add all alter and any array arraylen as asc assert assign at attributes audit authorization avg base_table begin between binary_integer body boolean by case cast char char_base check close cluster clusters colauth column comment commit compress connect connected constant constraint crash create current currval cursor data_base database date dba deallocate debugoff debugon decimal declare default definition delay delete desc digits dispose distinct do drop else elseif elsif enable end entry escape exception exception_init exchange exclusive exists exit external fast fetch file for force form from function generic goto grant group having identified if immediate in increment index indexes indicator initial initrans insert interface intersect into is key level library like limited local lock log logging long loop master maxextents maxtrans member minextents minus mislabel mode modify multiset new next no noaudit nocompress nologging noparallel not nowait number_base object of off offline on online only open option or order out package parallel partition pctfree pctincrease pctused pls_integer positive positiven pragma primary prior private privileges procedure public raise range raw read rebuild record ref references refresh release rename replace resource restrict return returning returns reverse revoke rollback row rowid rowlabel rownum rows run savepoint schema segment select separate session set share snapshot some space split sql start statement storage subtype successful synonym tabauth table tables tablespace task terminate then to trigger truncate type union unique unlimited unrecoverable unusable update use using validate value values variable view views when whenever where while with work"),
				builtin: r("abs acos add_months ascii asin atan atan2 average bfile bfilename bigserial bit blob ceil character chartorowid chr clob concat convert cos cosh count dec decode deref dual dump dup_val_on_index empty error exp false float floor found glb greatest hextoraw initcap instr instrb int integer isopen last_day least length lengthb ln lower lpad ltrim lub make_ref max min mlslabel mod months_between natural naturaln nchar nclob new_time next_day nextval nls_charset_decl_len nls_charset_id nls_charset_name nls_initcap nls_lower nls_sort nls_upper nlssort no_data_found notfound null number numeric nvarchar2 nvl others power rawtohex real reftohex round rowcount rowidtochar rowtype rpad rtrim serial sign signtype sin sinh smallint soundex sqlcode sqlerrm sqrt stddev string substr substrb sum sysdate tan tanh to_char text to_date to_label to_multi_byte to_number to_single_byte translate true trunc uid unlogged upper user userenv varchar varchar2 variance varying vsize xml"),
				operatorChars: /^[*+\-%<>!=~]/,
				dateSQL: r("date time timestamp"),
				support: r("doubleQuote nCharCast zerolessFloat binaryNumber hexNumber")
			}),
			e.defineMIME("text/x-hive", {
				name: "sql",
				keywords: r("select alter $elem$ $key$ $value$ add after all analyze and archive as asc before between binary both bucket buckets by cascade case cast change cluster clustered clusterstatus collection column columns comment compute concatenate continue create cross cursor data database databases dbproperties deferred delete delimited desc describe directory disable distinct distribute drop else enable end escaped exclusive exists explain export extended external false fetch fields fileformat first format formatted from full function functions grant group having hold_ddltime idxproperties if import in index indexes inpath inputdriver inputformat insert intersect into is items join keys lateral left like limit lines load local location lock locks mapjoin materialized minus msck no_drop nocompress not of offline on option or order out outer outputdriver outputformat overwrite partition partitioned partitions percent plus preserve procedure purge range rcfile read readonly reads rebuild recordreader recordwriter recover reduce regexp rename repair replace restrict revoke right rlike row schema schemas semi sequencefile serde serdeproperties set shared show show_database sort sorted ssl statistics stored streamtable table tables tablesample tblproperties temporary terminated textfile then tmp to touch transform trigger true unarchive undo union uniquejoin unlock update use using utc utc_tmestamp view when where while with"),
				builtin: r("bool boolean long timestamp tinyint smallint bigint int float double date datetime unsigned string array struct map uniontype"),
				atoms: r("false true null unknown"),
				operatorChars: /^[*+\-%<>!=]/,
				dateSQL: r("date timestamp"),
				support: r("ODBCdotTable doubleQuote binaryNumber hexNumber")
			}),
			e.defineMIME("text/x-pgsql", {
				name: "sql",
				client: r("source"),
				keywords: r(o + "a abort abs absent absolute access according action ada add admin after aggregate all allocate also always analyse analyze any are array array_agg array_max_cardinality asensitive assertion assignment asymmetric at atomic attribute attributes authorization avg backward base64 before begin begin_frame begin_partition bernoulli binary bit_length blob blocked bom both breadth c cache call called cardinality cascade cascaded case cast catalog catalog_name ceil ceiling chain characteristics characters character_length character_set_catalog character_set_name character_set_schema char_length check checkpoint class class_origin clob close cluster coalesce cobol collate collation collation_catalog collation_name collation_schema collect column columns column_name command_function command_function_code comment comments commit committed concurrently condition condition_number configuration conflict connect connection connection_name constraint constraints constraint_catalog constraint_name constraint_schema constructor contains content continue control conversion convert copy corr corresponding cost covar_pop covar_samp cross csv cube cume_dist current current_catalog current_date current_default_transform_group current_path current_role current_row current_schema current_time current_timestamp current_transform_group_for_type current_user cursor cursor_name cycle data database datalink datetime_interval_code datetime_interval_precision day db deallocate dec declare default defaults deferrable deferred defined definer degree delimiter delimiters dense_rank depth deref derived describe descriptor deterministic diagnostics dictionary disable discard disconnect dispatch dlnewcopy dlpreviouscopy dlurlcomplete dlurlcompleteonly dlurlcompletewrite dlurlpath dlurlpathonly dlurlpathwrite dlurlscheme dlurlserver dlvalue do document domain dynamic dynamic_function dynamic_function_code each element else empty enable encoding encrypted end end-exec end_frame end_partition enforced enum equals escape event every except exception exclude excluding exclusive exec execute exists exp explain expression extension external extract false family fetch file filter final first first_value flag float floor following for force foreign fortran forward found frame_row free freeze fs full function functions fusion g general generated get global go goto grant granted greatest grouping groups handler header hex hierarchy hold hour id identity if ignore ilike immediate immediately immutable implementation implicit import including increment indent index indexes indicator inherit inherits initially inline inner inout input insensitive instance instantiable instead integrity intersect intersection invoker isnull isolation k key key_member key_type label lag language large last last_value lateral lead leading leakproof least left length level library like_regex link listen ln load local localtime localtimestamp location locator lock locked logged lower m map mapping match matched materialized max maxvalue max_cardinality member merge message_length message_octet_length message_text method min minute minvalue mod mode modifies module month more move multiset mumps name names namespace national natural nchar nclob nesting new next nfc nfd nfkc nfkd nil no none normalize normalized nothing notify notnull nowait nth_value ntile null nullable nullif nulls number object occurrences_regex octets octet_length of off offset oids old only open operator option options ordering ordinality others out outer output over overlaps overlay overriding owned owner p pad parameter parameter_mode parameter_name parameter_ordinal_position parameter_specific_catalog parameter_specific_name parameter_specific_schema parser partial partition pascal passing passthrough password percent percentile_cont percentile_disc percent_rank period permission placing plans pli policy portion position position_regex power precedes preceding prepare prepared preserve primary prior privileges procedural procedure program public quote range rank read reads reassign recheck recovery recursive ref references referencing refresh regr_avgx regr_avgy regr_count regr_intercept regr_r2 regr_slope regr_sxx regr_sxy regr_syy reindex relative release rename repeatable replace replica requiring reset respect restart restore restrict result return returned_cardinality returned_length returned_octet_length returned_sqlstate returning returns revoke right role rollback rollup routine routine_catalog routine_name routine_schema row rows row_count row_number rule savepoint scale schema schema_name scope scope_catalog scope_name scope_schema scroll search second section security selective self sensitive sequence sequences serializable server server_name session session_user setof sets share show similar simple size skip snapshot some source space specific specifictype specific_name sql sqlcode sqlerror sqlexception sqlstate sqlwarning sqrt stable standalone start state statement static statistics stddev_pop stddev_samp stdin stdout storage strict strip structure style subclass_origin submultiset substring substring_regex succeeds sum symmetric sysid system system_time system_user t tables tablesample tablespace table_name temp template temporary then ties timezone_hour timezone_minute to token top_level_count trailing transaction transactions_committed transactions_rolled_back transaction_active transform transforms translate translate_regex translation treat trigger trigger_catalog trigger_name trigger_schema trim trim_array true truncate trusted type types uescape unbounded uncommitted under unencrypted unique unknown unlink unlisten unlogged unnamed unnest until untyped upper uri usage user user_defined_type_catalog user_defined_type_code user_defined_type_name user_defined_type_schema using vacuum valid validate validator value value_of varbinary variadic var_pop var_samp verbose version versioning view views volatile when whenever whitespace width_bucket window within work wrapper write xmlagg xmlattributes xmlbinary xmlcast xmlcomment xmlconcat xmldeclaration xmldocument xmlelement xmlexists xmlforest xmliterate xmlnamespaces xmlparse xmlpi xmlquery xmlroot xmlschema xmlserialize xmltable xmltext xmlvalidate year yes loop repeat"),
				builtin: r("bigint int8 bigserial serial8 bit varying varbit boolean bool box bytea character char varchar cidr circle date double precision float8 inet integer int int4 interval json jsonb line lseg macaddr money numeric decimal path pg_lsn point polygon real float4 smallint int2 smallserial serial2 serial serial4 text time without zone with timetz timestamp timestamptz tsquery tsvector txid_snapshot uuid xml"),
				atoms: r("false true null unknown"),
				operatorChars: /^[*+\-%<>!=&|^]/,
				dateSQL: r("date time timestamp"),
				support: r("ODBCdotTable decimallessFloat zerolessFloat binaryNumber hexNumber nCharCast charsetCast commentHash commentSpaceRequired")
			})
		} ()
	}),
	function(e) {
		"object" == typeof exports && "object" == typeof module ? e(require("../../lib/codemirror")) : "function" == typeof define && define.amd ? define(["../../lib/codemirror"], e) : e(CodeMirror)
	} (function(e) {
		"use strict";
		var t = {
			autoSelfClosers: {
				area: !0,
				base: !0,
				br: !0,
				col: !0,
				command: !0,
				embed: !0,
				frame: !0,
				hr: !0,
				img: !0,
				input: !0,
				keygen: !0,
				link: !0,
				meta: !0,
				param: !0,
				source: !0,
				track: !0,
				wbr: !0,
				menuitem: !0
			},
			implicitlyClosed: {
				dd: !0,
				li: !0,
				optgroup: !0,
				option: !0,
				p: !0,
				rp: !0,
				rt: !0,
				tbody: !0,
				td: !0,
				tfoot: !0,
				th: !0,
				tr: !0
			},
			contextGrabbers: {
				dd: {
					dd: !0,
					dt: !0
				},
				dt: {
					dd: !0,
					dt: !0
				},
				li: {
					li: !0
				},
				option: {
					option: !0,
					optgroup: !0
				},
				optgroup: {
					optgroup: !0
				},
				p: {
					address: !0,
					article: !0,
					aside: !0,
					blockquote: !0,
					dir: !0,
					div: !0,
					dl: !0,
					fieldset: !0,
					footer: !0,
					form: !0,
					h1: !0,
					h2: !0,
					h3: !0,
					h4: !0,
					h5: !0,
					h6: !0,
					header: !0,
					hgroup: !0,
					hr: !0,
					menu: !0,
					nav: !0,
					ol: !0,
					p: !0,
					pre: !0,
					section: !0,
					table: !0,
					ul: !0
				},
				rp: {
					rp: !0,
					rt: !0
				},
				rt: {
					rp: !0,
					rt: !0
				},
				tbody: {
					tbody: !0,
					tfoot: !0
				},
				td: {
					td: !0,
					th: !0
				},
				tfoot: {
					tbody: !0
				},
				th: {
					td: !0,
					th: !0
				},
				thead: {
					tbody: !0,
					tfoot: !0
				},
				tr: {
					tr: !0
				}
			},
			doNotIndent: {
				pre: !0
			},
			allowUnquoted: !0,
			allowMissing: !0,
			caseFold: !0
		},
		n = {
			autoSelfClosers: {},
			implicitlyClosed: {},
			contextGrabbers: {},
			doNotIndent: {},
			allowUnquoted: !1,
			allowMissing: !1,
			caseFold: !1
		};
		e.defineMode("xml",
		function(i, r) {
			function o(e, t) {
				function n(n) {
					return t.tokenize = n,
					n(e, t)
				}
				var i = e.next();
				if ("<" == i) return e.eat("!") ? e.eat("[") ? e.match("CDATA[") ? n(l("atom", "]]>")) : null: e.match("--") ? n(l("comment", "--\x3e")) : e.match("DOCTYPE", !0, !0) ? (e.eatWhile(/[\w\._\-]/), n(c(1))) : null: e.eat("?") ? (e.eatWhile(/[\w\._\-]/), t.tokenize = l("meta", "?>"), "meta") : (M = e.eat("/") ? "closeTag": "openTag", t.tokenize = a, "tag bracket");
				if ("&" == i) {
					return (e.eat("#") ? e.eat("x") ? e.eatWhile(/[a-fA-F\d]/) && e.eat(";") : e.eatWhile(/[\d]/) && e.eat(";") : e.eatWhile(/[\w\.\-:]/) && e.eat(";")) ? "atom": "error"
				}
				return e.eatWhile(/[^&<]/),
				null
			}
			function a(e, t) {
				var n = e.next();
				if (">" == n || "/" == n && e.eat(">")) return t.tokenize = o,
				M = ">" == n ? "endTag": "selfcloseTag",
				"tag bracket";
				if ("=" == n) return M = "equals",
				null;
				if ("<" == n) {
					t.tokenize = o,
					t.state = f,
					t.tagName = t.tagStart = null;
					var i = t.tokenize(e, t);
					return i ? i + " tag error": "tag error"
				}
				return /[\'\"]/.test(n) ? (t.tokenize = s(n), t.stringStartCol = e.column(), t.tokenize(e, t)) : (e.match(/^[^\s\u00a0=<>\"\']*[^\s\u00a0=<>\"\'\/]/), "word")
			}
			function s(e) {
				var t = function(t, n) {
					for (; ! t.eol();) if (t.next() == e) {
						n.tokenize = a;
						break
					}
					return "string"
				};
				return t.isInAttribute = !0,
				t
			}
			function l(e, t) {
				return function(n, i) {
					for (; ! n.eol();) {
						if (n.match(t)) {
							i.tokenize = o;
							break
						}
						n.next()
					}
					return e
				}
			}
			function c(e) {
				return function(t, n) {
					for (var i; null != (i = t.next());) {
						if ("<" == i) return n.tokenize = c(e + 1),
						n.tokenize(t, n);
						if (">" == i) {
							if (1 == e) {
								n.tokenize = o;
								break
							}
							return n.tokenize = c(e - 1),
							n.tokenize(t, n)
						}
					}
					return "meta"
				}
			}
			function u(e, t, n) {
				this.prev = e.context,
				this.tagName = t,
				this.indent = e.indented,
				this.startOfLine = n,
				(_.doNotIndent.hasOwnProperty(t) || e.context && e.context.noIndent) && (this.noIndent = !0)
			}
			function d(e) {
				e.context && (e.context = e.context.prev)
			}
			function h(e, t) {
				for (var n;;) {
					if (!e.context) return;
					if (n = e.context.tagName, !_.contextGrabbers.hasOwnProperty(n) || !_.contextGrabbers[n].hasOwnProperty(t)) return;
					d(e)
				}
			}
			function f(e, t, n) {
				return "openTag" == e ? (n.tagStart = t.column(), p) : "closeTag" == e ? m: f
			}
			function p(e, t, n) {
				return "word" == e ? (n.tagName = t.current(), T = "tag", y) : (T = "error", p)
			}
			function m(e, t, n) {
				if ("word" == e) {
					var i = t.current();
					return n.context && n.context.tagName != i && _.implicitlyClosed.hasOwnProperty(n.context.tagName) && d(n),
					n.context && n.context.tagName == i || !1 === _.matchClosing ? (T = "tag", g) : (T = "tag error", v)
				}
				return T = "error",
				v
			}
			function g(e, t, n) {
				return "endTag" != e ? (T = "error", g) : (d(n), f)
			}
			function v(e, t, n) {
				return T = "error",
				g(e, t, n)
			}
			function y(e, t, n) {
				if ("word" == e) return T = "attribute",
				b;
				if ("endTag" == e || "selfcloseTag" == e) {
					var i = n.tagName,
					r = n.tagStart;
					return n.tagName = n.tagStart = null,
					"selfcloseTag" == e || _.autoSelfClosers.hasOwnProperty(i) ? h(n, i) : (h(n, i), n.context = new u(n, i, r == n.indented)),
					f
				}
				return T = "error",
				y
			}
			function b(e, t, n) {
				return "equals" == e ? w: (_.allowMissing || (T = "error"), y(e, t, n))
			}
			function w(e, t, n) {
				return "string" == e ? k: "word" == e && _.allowUnquoted ? (T = "string", y) : (T = "error", y(e, t, n))
			}
			function k(e, t, n) {
				return "string" == e ? k: y(e, t, n)
			}
			var x = i.indentUnit,
			_ = {},
			C = r.htmlMode ? t: n;
			for (var S in C) _[S] = C[S];
			for (var S in r) _[S] = r[S];
			var M, T;
			return o.isInText = !0,
			{
				startState: function(e) {
					var t = {
						tokenize: o,
						state: f,
						indented: e || 0,
						tagName: null,
						tagStart: null,
						context: null
					};
					return null != e && (t.baseIndent = e),
					t
				},
				token: function(e, t) {
					if (!t.tagName && e.sol() && (t.indented = e.indentation()), e.eatSpace()) return null;
					M = null;
					var n = t.tokenize(e, t);
					return (n || M) && "comment" != n && (T = null, t.state = t.state(M || n, e, t), T && (n = "error" == T ? n + " error": T)),
					n
				},
				indent: function(t, n, i) {
					var r = t.context;
					if (t.tokenize.isInAttribute) return t.tagStart == t.indented ? t.stringStartCol + 1 : t.indented + x;
					if (r && r.noIndent) return e.Pass;
					if (t.tokenize != a && t.tokenize != o) return i ? i.match(/^(\s*)/)[0].length: 0;
					if (t.tagName) return ! 1 !== _.multilineTagIndentPastTag ? t.tagStart + t.tagName.length + 2 : t.tagStart + x * (_.multilineTagIndentFactor || 1);
					if (_.alignCDATA && /<!\[CDATA\[/.test(n)) return 0;
					var s = n && /^<(\/)?([\w_:\.-]*)/.exec(n);
					if (s && s[1]) for (; r;) {
						if (r.tagName == s[2]) {
							r = r.prev;
							break
						}
						if (!_.implicitlyClosed.hasOwnProperty(r.tagName)) break;
						r = r.prev
					} else if (s) for (; r;) {
						var l = _.contextGrabbers[r.tagName];
						if (!l || !l.hasOwnProperty(s[2])) break;
						r = r.prev
					}
					for (; r && r.prev && !r.startOfLine;) r = r.prev;
					return r ? r.indent + x: t.baseIndent || 0
				},
				electricInput: /<\/[\s\w:]+>$/,
				blockCommentStart: "\x3c!--",
				blockCommentEnd: "--\x3e",
				configuration: _.htmlMode ? "html": "xml",
				helperType: _.htmlMode ? "html": "xml",
				skipAttribute: function(e) {
					e.state == w && (e.state = y)
				}
			}
		}),
		e.defineMIME("text/xml", "xml"),
		e.defineMIME("application/xml", "xml"),
		e.mimeModes.hasOwnProperty("text/html") || e.defineMIME("text/html", {
			name: "xml",
			htmlMode: !0
		})
	}),
	function(e) {
		"object" == typeof exports && "object" == typeof module ? e(require("../../lib/codemirror")) : "function" == typeof define && define.amd ? define(["../../lib/codemirror"], e) : e(CodeMirror)
	} (function(e) {
		"use strict";
		e.defineMode("xquery",
		function() {
			function e(e, t, n) {
				return t.tokenize = n,
				n(e, t)
			}
			function t(t, a) {
				var d = t.next(),
				f = !1,
				m = p(t);
				if ("<" == d) {
					if (t.match("!--", !0)) return e(t, a, s);
					if (t.match("![CDATA", !1)) return a.tokenize = l,
					"tag";
					if (t.match("?", !1)) return e(t, a, c);
					var b = t.eat("/");
					t.eatSpace();
					for (var w, k = ""; w = t.eat(/[^\s\u00a0=<>\"\'\/?]/);) k += w;
					return e(t, a, o(k, b))
				}
				if ("{" == d) return g(a, {
					type: "codeblock"
				}),
				null;
				if ("}" == d) return v(a),
				null;
				if (u(a)) return ">" == d ? "tag": "/" == d && t.eat(">") ? (v(a), "tag") : "variable";
				if (/\d/.test(d)) return t.match(/^\d*(?:\.\d*)?(?:E[+\-]?\d+)?/),
				"atom";
				if ("(" === d && t.eat(":")) return g(a, {
					type: "comment"
				}),
				e(t, a, n);
				if (m || '"' !== d && "'" !== d) {
					if ("$" === d) return e(t, a, r);
					if (":" === d && t.eat("=")) return "keyword";
					if ("(" === d) return g(a, {
						type: "paren"
					}),
					null;
					if (")" === d) return v(a),
					null;
					if ("[" === d) return g(a, {
						type: "bracket"
					}),
					null;
					if ("]" === d) return v(a),
					null;
					var x = y.propertyIsEnumerable(d) && y[d];
					if (m && '"' === d) for (;
					'"' !== t.next(););
					if (m && "'" === d) for (;
					"'" !== t.next(););
					x || t.eatWhile(/[\w\$_-]/);
					var _ = t.eat(":"); ! t.eat(":") && _ && t.eatWhile(/[\w\$_-]/),
					t.match(/^[ \t]*\(/, !1) && (f = !0);
					var C = t.current();
					return x = y.propertyIsEnumerable(C) && y[C],
					f && !x && (x = {
						type: "function_call",
						style: "variable def"
					}),
					h(a) ? (v(a), "variable") : (("element" == C || "attribute" == C || "axis_specifier" == x.type) && g(a, {
						type: "xmlconstructor"
					}), x ? x.style: "variable")
				}
				return e(t, a, i(d))
			}
			function n(e, t) {
				for (var n, i = !1,
				r = !1,
				o = 0; n = e.next();) {
					if (")" == n && i) {
						if (! (o > 0)) {
							v(t);
							break
						}
						o--
					} else ":" == n && r && o++;
					i = ":" == n,
					r = "(" == n
				}
				return "comment"
			}
			function i(e, n) {
				return function(r, o) {
					var a;
					if (f(o) && r.current() == e) return v(o),
					n && (o.tokenize = n),
					"string";
					if (g(o, {
						type: "string",
						name: e,
						tokenize: i(e, n)
					}), r.match("{", !1) && d(o)) return o.tokenize = t,
					"string";
					for (; a = r.next();) {
						if (a == e) {
							v(o),
							n && (o.tokenize = n);
							break
						}
						if (r.match("{", !1) && d(o)) return o.tokenize = t,
						"string"
					}
					return "string"
				}
			}
			function r(e, n) {
				var i = /[\w\$_-]/;
				if (e.eat('"')) {
					for (;
					'"' !== e.next(););
					e.eat(":")
				} else e.eatWhile(i),
				e.match(":=", !1) || e.eat(":");
				return e.eatWhile(i),
				n.tokenize = t,
				"variable"
			}
			function o(e, n) {
				return function(i, r) {
					return i.eatSpace(),
					n && i.eat(">") ? (v(r), r.tokenize = t, "tag") : (i.eat("/") || g(r, {
						type: "tag",
						name: e,
						tokenize: t
					}), i.eat(">") ? (r.tokenize = t, "tag") : (r.tokenize = a, "tag"))
				}
			}
			function a(n, r) {
				var o = n.next();
				return "/" == o && n.eat(">") ? (d(r) && v(r), u(r) && v(r), "tag") : ">" == o ? (d(r) && v(r), "tag") : "=" == o ? null: '"' == o || "'" == o ? e(n, r, i(o, a)) : (d(r) || g(r, {
					type: "attribute",
					tokenize: a
				}), n.eat(/[a-zA-Z_:]/), n.eatWhile(/[-a-zA-Z0-9_:.]/), n.eatSpace(), (n.match(">", !1) || n.match("/", !1)) && (v(r), r.tokenize = t), "attribute")
			}
			function s(e, n) {
				for (var i; i = e.next();) if ("-" == i && e.match("->", !0)) return n.tokenize = t,
				"comment"
			}
			function l(e, n) {
				for (var i; i = e.next();) if ("]" == i && e.match("]", !0)) return n.tokenize = t,
				"comment"
			}
			function c(e, n) {
				for (var i; i = e.next();) if ("?" == i && e.match(">", !0)) return n.tokenize = t,
				"comment meta"
			}
			function u(e) {
				return m(e, "tag")
			}
			function d(e) {
				return m(e, "attribute")
			}
			function h(e) {
				return m(e, "xmlconstructor")
			}
			function f(e) {
				return m(e, "string")
			}
			function p(e) {
				return '"' === e.current() ? e.match(/^[^\"]+\"\:/, !1) : "'" === e.current() && e.match(/^[^\"]+\'\:/, !1)
			}
			function m(e, t) {
				return e.stack.length && e.stack[e.stack.length - 1].type == t
			}
			function g(e, t) {
				e.stack.push(t)
			}
			function v(e) {
				e.stack.pop();
				var n = e.stack.length && e.stack[e.stack.length - 1].tokenize;
				e.tokenize = n || t
			}
			var y = function() {
				function e(e) {
					return {
						type: e,
						style: "keyword"
					}
				}
				for (var t = e("keyword a"), n = e("keyword b"), i = e("keyword c"), r = e("operator"), o = {
					type: "atom",
					style: "atom"
				},
				a = {
					type: "axis_specifier",
					style: "qualifier"
				},
				s = {
					if: t,
					switch: t,
					while: t,
					for: t,
					else: n,
					then: n,
					try: n,
					finally: n,
					catch: n,
					element: i,
					attribute: i,
					let: i,
					implements: i,
					import: i,
					module: i,
					namespace: i,
					return: i,
					super: i,
					this: i,
					throws: i,
					where: i,
					private: i,
					",": {
						type: "punctuation",
						style: null
					},
					null: o,
					"fn:false()": o,
					"fn:true()": o
				},
				l = ["after", "ancestor", "ancestor-or-self", "and", "as", "ascending", "assert", "attribute", "before", "by", "case", "cast", "child", "comment", "declare", "default", "define", "descendant", "descendant-or-self", "descending", "document", "document-node", "element", "else", "eq", "every", "except", "external", "following", "following-sibling", "follows", "for", "function", "if", "import", "in", "instance", "intersect", "item", "let", "module", "namespace", "node", "node", "of", "only", "or", "order", "parent", "precedes", "preceding", "preceding-sibling", "processing-instruction", "ref", "return", "returns", "satisfies", "schema", "schema-element", "self", "some", "sortby", "stable", "text", "then", "to", "treat", "typeswitch", "union", "variable", "version", "where", "xquery", "empty-sequence"], c = 0, u = l.length; u > c; c++) s[l[c]] = e(l[c]);
				for (var d = ["xs:string", "xs:float", "xs:decimal", "xs:double", "xs:integer", "xs:boolean", "xs:date", "xs:dateTime", "xs:time", "xs:duration", "xs:dayTimeDuration", "xs:time", "xs:yearMonthDuration", "numeric", "xs:hexBinary", "xs:base64Binary", "xs:anyURI", "xs:QName", "xs:byte", "xs:boolean", "xs:anyURI", "xf:yearMonthDuration"], c = 0, u = d.length; u > c; c++) s[d[c]] = o;
				for (var h = ["eq", "ne", "lt", "le", "gt", "ge", ":=", "=", ">", ">=", "<", "<=", ".", "|", "?", "and", "or", "div", "idiv", "mod", "*", "/", "+", "-"], c = 0, u = h.length; u > c; c++) s[h[c]] = r;
				for (var f = ["self::", "attribute::", "child::", "descendant::", "descendant-or-self::", "parent::", "ancestor::", "ancestor-or-self::", "following::", "preceding::", "following-sibling::", "preceding-sibling::"], c = 0, u = f.length; u > c; c++) s[f[c]] = a;
				return s
			} ();
			return {
				startState: function() {
					return {
						tokenize: t,
						cc: [],
						stack: []
					}
				},
				token: function(e, t) {
					return e.eatSpace() ? null: t.tokenize(e, t)
				},
				blockCommentStart: "(:",
				blockCommentEnd: ":)"
			}
		}),
		e.defineMIME("application/xquery", "xquery")
	}),
	function(e) {
		"object" == typeof exports && "object" == typeof module ? e(require("../../lib/codemirror")) : "function" == typeof define && define.amd ? define(["../../lib/codemirror"], e) : e(CodeMirror)
	} (function(e) {
		"use strict";
		e.defineMode("yaml",
		function() {
			var e = ["true", "false", "on", "off", "yes", "no"],
			t = new RegExp("\\b((" + e.join(")|(") + "))$", "i");
			return {
				token: function(e, n) {
					var i = e.peek(),
					r = n.escaped;
					if (n.escaped = !1, "#" == i && (0 == e.pos || /\s/.test(e.string.charAt(e.pos - 1)))) return e.skipToEnd(),
					"comment";
					if (e.match(/^('([^']|\\.)*'?|"([^"]|\\.)*"?)/)) return "string";
					if (n.literal && e.indentation() > n.keyCol) return e.skipToEnd(),
					"string";
					if (n.literal && (n.literal = !1), e.sol()) {
						if (n.keyCol = 0, n.pair = !1, n.pairStart = !1, e.match(/---/)) return "def";
						if (e.match(/\.\.\./)) return "def";
						if (e.match(/\s*-\s+/)) return "meta"
					}
					if (e.match(/^(\{|\}|\[|\])/)) return "{" == i ? n.inlinePairs++:"}" == i ? n.inlinePairs--:"[" == i ? n.inlineList++:n.inlineList--,
					"meta";
					if (n.inlineList > 0 && !r && "," == i) return e.next(),
					"meta";
					if (n.inlinePairs > 0 && !r && "," == i) return n.keyCol = 0,
					n.pair = !1,
					n.pairStart = !1,
					e.next(),
					"meta";
					if (n.pairStart) {
						if (e.match(/^\s*(\||\>)\s*/)) return n.literal = !0,
						"meta";
						if (e.match(/^\s*(\&|\*)[a-z0-9\._-]+\b/i)) return "variable-2";
						if (0 == n.inlinePairs && e.match(/^\s*-?[0-9\.\,]+\s?$/)) return "number";
						if (n.inlinePairs > 0 && e.match(/^\s*-?[0-9\.\,]+\s?(?=(,|}))/)) return "number";
						if (e.match(t)) return "keyword"
					}
					return ! n.pair && e.match(/^\s*(?:[,\[\]{}&*!|>'"%@`][^\s'":]|[^,\[\]{}#&*!|>'"%@`])[^#]*?(?=\s*:($|\s))/) ? (n.pair = !0, n.keyCol = e.indentation(), "atom") : n.pair && e.match(/^:\s*/) ? (n.pairStart = !0, "meta") : (n.pairStart = !1, n.escaped = "\\" == i, e.next(), null)
				},
				startState: function() {
					return {
						pair: !1,
						pairStart: !1,
						keyCol: 0,
						inlinePairs: 0,
						inlineList: 0,
						literal: !1,
						escaped: !1
					}
				}
			}
		}),
		e.defineMIME("text/x-yaml", "yaml")
	}),
	function(e) {
		"object" == typeof exports && "object" == typeof module ? e(require("../../lib/codemirror")) : "function" == typeof define && define.amd ? define(["../../lib/codemirror"], e) : e(CodeMirror)
	} (function(e) {
		"use strict";
		function t(e) {
			for (var t = 0; t < e.state.activeLines.length; t++) e.removeLineClass(e.state.activeLines[t], "wrap", o),
			e.removeLineClass(e.state.activeLines[t], "background", a),
			e.removeLineClass(e.state.activeLines[t], "gutter", s)
		}
		function n(e, t) {
			if (e.length != t.length) return ! 1;
			for (var n = 0; n < e.length; n++) if (e[n] != t[n]) return ! 1;
			return ! 0
		}
		function i(e, i) {
			for (var r = [], l = 0; l < i.length; l++) {
				var c = i[l];
				if (c.empty()) {
					var u = e.getLineHandleVisualStart(c.head.line);
					r[r.length - 1] != u && r.push(u)
				}
			}
			n(e.state.activeLines, r) || e.operation(function() {
				t(e);
				for (var n = 0; n < r.length; n++) e.addLineClass(r[n], "wrap", o),
				e.addLineClass(r[n], "background", a),
				e.addLineClass(r[n], "gutter", s);
				e.state.activeLines = r
			})
		}
		function r(e, t) {
			i(e, t.ranges)
		}
		var o = "CodeMirror-activeline",
		a = "CodeMirror-activeline-background",
		s = "CodeMirror-activeline-gutter";
		e.defineOption("styleActiveLine", !1,
		function(n, o, a) {
			var s = a && a != e.Init;
			o && !s ? (n.state.activeLines = [], i(n, n.listSelections()), n.on("beforeSelectionChange", r)) : !o && s && (n.off("beforeSelectionChange", r), t(n), delete n.state.activeLines)
		})
	}),
	function(e) {
		"object" == typeof exports && "object" == typeof module ? e(require("../../lib/codemirror")) : "function" == typeof define && define.amd ? define(["../../lib/codemirror"], e) : e(CodeMirror)
	} (function(e) {
		"use strict";
		var t = /[\w$]+/;
		e.registerHelper("hint", "anyword",
		function(n, i) {
			for (var r = i && i.word || t,
			o = i && i.range || 500,
			a = n.getCursor(), s = n.getLine(a.line), l = a.ch, c = l; c && r.test(s.charAt(c - 1));)--c;
			for (var u = c != l && s.slice(c, l), d = i && i.list || [], h = {},
			f = new RegExp(r.source, "g"), p = -1; 1 >= p; p += 2) for (var m = a.line,
			g = Math.min(Math.max(m + p * o, n.firstLine()), n.lastLine()) + p; m != g; m += p) for (var v, y = n.getLine(m); v = f.exec(y);)(m != a.line || v[0] !== u) && (u && 0 != v[0].lastIndexOf(u, 0) || Object.prototype.hasOwnProperty.call(h, v[0]) || (h[v[0]] = !0, d.push(v[0])));
			return {
				list: d,
				from: e.Pos(a.line, c),
				to: e.Pos(a.line, l)
			}
		})
	}),
	function(e) {
		"object" == typeof exports && "object" == typeof module ? e(require("../../lib/codemirror")) : "function" == typeof define && define.amd ? define(["../../lib/codemirror"], e) : e(CodeMirror)
	} (function(e) {
		function t(e, t) {
			return "pairs" == t && "string" == typeof e ? e: "object" == typeof e && null != e[t] ? e[t] : l[t]
		}
		function n(e) {
			var t = e.state.closeBrackets;
			return t ? e.getModeAt(e.getCursor()).closeBrackets || t: null
		}
		function i(t) {
			var n = e.cmpPos(t.anchor, t.head) > 0;
			return {
				anchor: new c(t.anchor.line, t.anchor.ch + (n ? -1 : 1)),
				head: new c(t.head.line, t.head.ch + (n ? 1 : -1))
			}
		}
		function r(r, a) {
			var l = n(r);
			if (!l || r.getOption("disableInput")) return e.Pass;
			var u = t(l, "pairs"),
			d = u.indexOf(a);
			if ( - 1 == d) return e.Pass;
			for (var h, f = t(l, "triples"), p = u.charAt(d + 1) == a, m = r.listSelections(), g = d % 2 == 0, v = 0; v < m.length; v++) {
				var y, b = m[v],
				w = b.head,
				k = r.getRange(w, c(w.line, w.ch + 1));
				if (g && !b.empty()) y = "surround";
				else if (!p && g || k != a) if (p && w.ch > 1 && f.indexOf(a) >= 0 && r.getRange(c(w.line, w.ch - 2), w) == a + a && (w.ch <= 2 || r.getRange(c(w.line, w.ch - 3), c(w.line, w.ch - 2)) != a)) y = "addFour";
				else if (p) {
					if (e.isWordChar(k) || !s(r, w, a)) return e.Pass;
					y = "both"
				} else {
					if (!g || r.getLine(w.line).length != w.ch && !o(k, u) && !/\s/.test(k)) return e.Pass;
					y = "both"
				} else y = f.indexOf(a) >= 0 && r.getRange(w, c(w.line, w.ch + 3)) == a + a + a ? "skipThree": "skip";
				if (h) {
					if (h != y) return e.Pass
				} else h = y
			}
			var x = d % 2 ? u.charAt(d - 1) : a,
			_ = d % 2 ? a: u.charAt(d + 1);
			r.operation(function() {
				if ("skip" == h) r.execCommand("goCharRight");
				else if ("skipThree" == h) for (t = 0; 3 > t; t++) r.execCommand("goCharRight");
				else if ("surround" == h) {
					for (var e = r.getSelections(), t = 0; t < e.length; t++) e[t] = x + e[t] + _;
					r.replaceSelections(e, "around"),
					e = r.listSelections().slice();
					for (t = 0; t < e.length; t++) e[t] = i(e[t]);
					r.setSelections(e)
				} else "both" == h ? (r.replaceSelection(x + _, null), r.triggerElectric(x + _), r.execCommand("goCharLeft")) : "addFour" == h && (r.replaceSelection(x + x + x + x, "before"), r.execCommand("goCharRight"))
			})
		}
		function o(e, t) {
			var n = t.lastIndexOf(e);
			return n > -1 && n % 2 == 1
		}
		function a(e, t) {
			var n = e.getRange(c(t.line, t.ch - 1), c(t.line, t.ch + 1));
			return 2 == n.length ? n: null
		}
		function s(t, n, i) {
			var r = t.getLine(n.line),
			o = t.getTokenAt(n);
			if (/\bstring2?\b/.test(o.type)) return ! 1;
			var a = new e.StringStream(r.slice(0, n.ch) + i + r.slice(n.ch), 4);
			for (a.pos = a.start = o.start;;) {
				var s = t.getMode().token(a, o.state);
				if (a.pos >= n.ch + 1) return /\bstring2?\b/.test(s);
				a.start = a.pos
			}
		}
		var l = {
			pairs: "()[]{}''\"\"",
			triples: "",
			explode: "[]{}"
		},
		c = e.Pos;
		e.defineOption("autoCloseBrackets", !1,
		function(t, n, i) {
			i && i != e.Init && (t.removeKeyMap(d), t.state.closeBrackets = null),
			n && (t.state.closeBrackets = n, t.addKeyMap(d))
		});
		for (var u = l.pairs + "`",
		d = {
			Backspace: function(i) {
				var r = n(i);
				if (!r || i.getOption("disableInput")) return e.Pass;
				for (var o = t(r, "pairs"), s = i.listSelections(), l = 0; l < s.length; l++) {
					if (!s[l].empty()) return e.Pass;
					var u = a(i, s[l].head);
					if (!u || o.indexOf(u) % 2 != 0) return e.Pass
				}
				for (l = s.length - 1; l >= 0; l--) {
					var d = s[l].head;
					i.replaceRange("", c(d.line, d.ch - 1), c(d.line, d.ch + 1), "+delete")
				}
			},
			Enter: function(i) {
				var r = n(i),
				o = r && t(r, "explode");
				if (!o || i.getOption("disableInput")) return e.Pass;
				for (var s = i.listSelections(), l = 0; l < s.length; l++) {
					if (!s[l].empty()) return e.Pass;
					var c = a(i, s[l].head);
					if (!c || o.indexOf(c) % 2 != 0) return e.Pass
				}
				i.operation(function() {
					i.replaceSelection("\n\n", null),
					i.execCommand("goCharLeft"),
					s = i.listSelections();
					for (var e = 0; e < s.length; e++) {
						var t = s[e].head.line;
						i.indentLine(t, null, !0),
						i.indentLine(t + 1, null, !0)
					}
				})
			}
		},
		h = 0; h < u.length; h++) d["'" + u.charAt(h) + "'"] = function(e) {
			return function(t) {
				return r(t, e)
			}
		} (u.charAt(h))
	}),
	function(e) {
		"object" == typeof exports && "object" == typeof module ? e(require("../../lib/codemirror"), require("../fold/xml-fold")) : "function" == typeof define && define.amd ? define(["../../lib/codemirror", "../fold/xml-fold"], e) : e(CodeMirror)
	} (function(e) {
		function t(t) {
			if (t.getOption("disableInput")) return e.Pass;
			for (var n = t.listSelections(), i = [], l = 0; l < n.length; l++) {
				if (!n[l].empty()) return e.Pass;
				var c = n[l].head,
				u = t.getTokenAt(c),
				d = e.innerMode(t.getMode(), u.state),
				h = d.state;
				if ("xml" != d.mode.name || !h.tagName) return e.Pass;
				var f = t.getOption("autoCloseTags"),
				p = "html" == d.mode.configuration,
				m = "object" == typeof f && f.dontCloseTags || p && a,
				g = "object" == typeof f && f.indentTags || p && s,
				v = h.tagName;
				u.end > c.ch && (v = v.slice(0, v.length - u.end + c.ch));
				var y = v.toLowerCase();
				if (!v || "string" == u.type && (u.end != c.ch || !/[\"\']/.test(u.string.charAt(u.string.length - 1)) || 1 == u.string.length) || "tag" == u.type && "closeTag" == h.type || u.string.indexOf("/") == u.string.length - 1 || m && r(m, y) > -1 || o(t, v, c, h, !0)) return e.Pass;
				var b = g && r(g, y) > -1;
				i[l] = {
					indent: b,
					text: ">" + (b ? "\n\n": "") + "</" + v + ">",
					newPos: b ? e.Pos(c.line + 1, 0) : e.Pos(c.line, c.ch + 1)
				}
			}
			for (l = n.length - 1; l >= 0; l--) {
				var w = i[l];
				t.replaceRange(w.text, n[l].head, n[l].anchor, "+insert");
				var k = t.listSelections().slice(0);
				k[l] = {
					head: w.newPos,
					anchor: w.newPos
				},
				t.setSelections(k),
				w.indent && (t.indentLine(w.newPos.line, null, !0), t.indentLine(w.newPos.line + 1, null, !0))
			}
		}
		function n(t, n) {
			for (var i = t.listSelections(), r = [], a = n ? "/": "</", s = 0; s < i.length; s++) {
				if (!i[s].empty()) return e.Pass;
				var l = i[s].head,
				c = t.getTokenAt(l),
				u = e.innerMode(t.getMode(), c.state),
				d = u.state;
				if (n && ("string" == c.type || "<" != c.string.charAt(0) || c.start != l.ch - 1)) return e.Pass;
				var h;
				if ("xml" != u.mode.name) if ("htmlmixed" == t.getMode().name && "javascript" == u.mode.name) h = a + "script";
				else {
					if ("htmlmixed" != t.getMode().name || "css" != u.mode.name) return e.Pass;
					h = a + "style"
				} else {
					if (!d.context || !d.context.tagName || o(t, d.context.tagName, l, d)) return e.Pass;
					h = a + d.context.tagName
				}
				">" != t.getLine(l.line).charAt(c.end) && (h += ">"),
				r[s] = h
			}
			t.replaceSelections(r),
			i = t.listSelections();
			for (s = 0; s < i.length; s++)(s == i.length - 1 || i[s].head.line < i[s + 1].head.line) && t.indentLine(i[s].head.line)
		}
		function i(t) {
			return t.getOption("disableInput") ? e.Pass: n(t, !0)
		}
		function r(e, t) {
			if (e.indexOf) return e.indexOf(t);
			for (var n = 0,
			i = e.length; i > n; ++n) if (e[n] == t) return n;
			return - 1
		}
		function o(t, n, i, r, o) {
			if (!e.scanForClosingTag) return ! 1;
			var a = Math.min(t.lastLine() + 1, i.line + 500),
			s = e.scanForClosingTag(t, i, null, a);
			if (!s || s.tag != n) return ! 1;
			for (var l = r.context,
			c = o ? 1 : 0; l && l.tagName == n; l = l.prev)++c;
			i = s.to;
			for (var u = 1; c > u; u++) {
				var d = e.scanForClosingTag(t, i, null, a);
				if (!d || d.tag != n) return ! 1;
				i = d.to
			}
			return ! 0
		}
		e.defineOption("autoCloseTags", !1,
		function(n, r, o) {
			if (o != e.Init && o && n.removeKeyMap("autoCloseTags"), r) {
				var a = {
					name: "autoCloseTags"
				}; ("object" != typeof r || r.whenClosing) && (a["'/'"] = function(e) {
					return i(e)
				}),
				("object" != typeof r || r.whenOpening) && (a["'>'"] = function(e) {
					return t(e)
				}),
				n.addKeyMap(a)
			}
		});
		var a = ["area", "base", "br", "col", "command", "embed", "hr", "img", "input", "keygen", "link", "meta", "param", "source", "track", "wbr"],
		s = ["applet", "blockquote", "body", "button", "div", "dl", "fieldset", "form", "frameset", "h1", "h2", "h3", "h4", "h5", "h6", "head", "html", "iframe", "layer", "legend", "object", "ol", "p", "select", "table", "ul"];
		e.commands.closeTag = function(e) {
			return n(e)
		}
	}),
	function(e) {
		"object" == typeof exports && "object" == typeof module ? e(require("../../lib/codemirror"), require("../../mode/css/css")) : "function" == typeof define && define.amd ? define(["../../lib/codemirror", "../../mode/css/css"], e) : e(CodeMirror)
	} (function(e) {
		"use strict";
		var t = {
			link: 1,
			visited: 1,
			active: 1,
			hover: 1,
			focus: 1,
			"first-letter": 1,
			"first-line": 1,
			"first-child": 1,
			before: 1,
			after: 1,
			lang: 1
		};
		e.registerHelper("hint", "css",
		function(n) {
			function i(e) {
				for (var t in e) c && 0 != t.lastIndexOf(c, 0) || d.push(t)
			}
			var r = n.getCursor(),
			o = n.getTokenAt(r),
			a = e.innerMode(n.getMode(), o.state);
			if ("css" == a.mode.name) {
				if ("keyword" == o.type && 0 == "!important".indexOf(o.string)) return {
					list: ["!important"],
					from: e.Pos(r.line, o.start),
					to: e.Pos(r.line, o.end)
				};
				var s = o.start,
				l = r.ch,
				c = o.string.slice(0, l - s);
				/[^\w$_-]/.test(c) && (c = "", s = l = r.ch);
				var u = e.resolveMode("text/css"),
				d = [],
				h = a.state.state;
				return "pseudo" == h || "variable-3" == o.type ? i(t) : "block" == h || "maybeprop" == h ? i(u.propertyKeywords) : "prop" == h || "parens" == h || "at" == h || "params" == h ? (i(u.valueKeywords), i(u.colorKeywords)) : ("media" == h || "media_parens" == h) && (i(u.mediaTypes), i(u.mediaFeatures)),
				d.length ? {
					list: d,
					from: e.Pos(r.line, s),
					to: e.Pos(r.line, l)
				}: void 0
			}
		})
	}),
	function(e) {
		"object" == typeof exports && "object" == typeof module ? e(require("../../lib/codemirror")) : "function" == typeof define && define.amd ? define(["../../lib/codemirror"], e) : e(CodeMirror)
	} (function(e) {
		"use strict";
		function t(e, t, n) {
			for (var i = n.paragraphStart || e.getHelper(t, "paragraphStart"), r = t.line, o = e.firstLine(); r > o; --r) {
				c = e.getLine(r);
				if (i && i.test(c)) break;
				if (!/\S/.test(c)) {++r;
					break
				}
			}
			for (var a = n.paragraphEnd || e.getHelper(t, "paragraphEnd"), s = t.line + 1, l = e.lastLine(); l >= s; ++s) {
				var c = e.getLine(s);
				if (a && a.test(c)) {++s;
					break
				}
				if (!/\S/.test(c)) break
			}
			return {
				from: r,
				to: s
			}
		}
		function n(e, t, n, i) {
			for (var r = t; r > 0 && !n.test(e.slice(r - 1, r + 1)); --r);
			for (var o = !0;; o = !1) {
				var a = r;
				if (i) for (;
				" " == e.charAt(a - 1);)--a;
				if (0 != a || !o) return {
					from: a,
					to: r
				};
				r = t
			}
		}
		function i(t, i, o, a) {
			i = t.clipPos(i),
			o = t.clipPos(o);
			var s = a.column || 80,
			l = a.wrapOn || /\s\S|-[^\.\d]/,
			c = !1 !== a.killTrailingSpace,
			u = [],
			d = "",
			h = i.line,
			f = t.getRange(i, o, !1);
			if (!f.length) return null;
			for (var p = f[0].match(/^[ \t]*/)[0], m = 0; m < f.length; ++m) {
				var g = f[m],
				v = d.length,
				y = 0;
				d && g && !l.test(d.charAt(d.length - 1) + g.charAt(0)) && (d += " ", y = 1);
				var b = "";
				if (m && (b = g.match(/^\s*/)[0], g = g.slice(b.length)), d += g, m) {
					var w = d.length > s && p == b && n(d, s, l, c);
					w && w.from == v && w.to == v + y ? (d = p + g, ++h) : u.push({
						text: [y ? " ": ""],
						from: r(h, v),
						to: r(h + 1, b.length)
					})
				}
				for (; d.length > s;) {
					var k = n(d, s, l, c);
					u.push({
						text: ["", p],
						from: r(h, k.from),
						to: r(h, k.to)
					}),
					d = p + d.slice(k.to),
					++h
				}
			}
			return u.length && t.operation(function() {
				for (var n = 0; n < u.length; ++n) {
					var i = u[n]; (i.text || e.cmpPos(i.from, i.to)) && t.replaceRange(i.text, i.from, i.to)
				}
			}),
			u.length ? {
				from: u[0].from,
				to: e.changeEnd(u[u.length - 1])
			}: null
		}
		var r = e.Pos;
		e.defineExtension("wrapParagraph",
		function(e, n) {
			n = n || {},
			e || (e = this.getCursor());
			var o = t(this, e, n);
			return i(this, r(o.from, 0), r(o.to - 1), n)
		}),
		e.commands.wrapLines = function(e) {
			e.operation(function() {
				for (var n = e.listSelections(), o = e.lastLine() + 1, a = n.length - 1; a >= 0; a--) {
					var s, l = n[a];
					if (l.empty()) {
						var c = t(e, l.head, {});
						s = {
							from: r(c.from, 0),
							to: r(c.to - 1)
						}
					} else s = {
						from: l.from(),
						to: l.to()
					};
					s.to.line >= o || (o = s.from.line, i(e, s.from, s.to, {}))
				}
			})
		},
		e.defineExtension("wrapRange",
		function(e, t, n) {
			return i(this, e, t, n || {})
		}),
		e.defineExtension("wrapParagraphsInRange",
		function(e, n, o) {
			o = o || {};
			for (var a = this,
			s = [], l = e.line; l <= n.line;) {
				var c = t(a, r(l, 0), o);
				s.push(c),
				l = c.to
			}
			var u = !1;
			return s.length && a.operation(function() {
				for (var e = s.length - 1; e >= 0; --e) u = u || i(a, r(s[e].from, 0), r(s[e].to - 1), o)
			}),
			u
		})
	}),
	function(e) {
		"object" == typeof exports && "object" == typeof module ? e(require("../../lib/codemirror"), require("./xml-hint")) : "function" == typeof define && define.amd ? define(["../../lib/codemirror", "./xml-hint"], e) : e(CodeMirror)
	} (function(e) {
		"use strict";
		function t(e) {
			for (var t in u) u.hasOwnProperty(t) && (e.attrs[t] = u[t])
		}
		var n = "ab aa af ak sq am ar an hy as av ae ay az bm ba eu be bn bh bi bs br bg my ca ch ce ny zh cv kw co cr hr cs da dv nl dz en eo et ee fo fj fi fr ff gl ka de el gn gu ht ha he hz hi ho hu ia id ie ga ig ik io is it iu ja jv kl kn kr ks kk km ki rw ky kv kg ko ku kj la lb lg li ln lo lt lu lv gv mk mg ms ml mt mi mr mh mn na nv nb nd ne ng nn no ii nr oc oj cu om or os pa pi fa pl ps pt qu rm rn ro ru sa sc sd se sm sg sr gd sn si sk sl so st es su sw ss sv ta te tg th ti bo tk tl tn to tr ts tt tw ty ug uk ur uz ve vi vo wa cy wo fy xh yi yo za zu".split(" "),
		i = ["_blank", "_self", "_top", "_parent"],
		r = ["ascii", "utf-8", "utf-16", "latin1", "latin1"],
		o = ["get", "post", "put", "delete"],
		a = ["application/x-www-form-urlencoded", "multipart/form-data", "text/plain"],
		s = ["all", "screen", "print", "embossed", "braille", "handheld", "print", "projection", "screen", "tty", "tv", "speech", "3d-glasses", "resolution [>][<][=] [X]", "device-aspect-ratio: X/Y", "orientation:portrait", "orientation:landscape", "device-height: [X]", "device-width: [X]"],
		l = {
			attrs: {}
		},
		c = {
			a: {
				attrs: {
					href: null,
					ping: null,
					type: null,
					media: s,
					target: i,
					hreflang: n
				}
			},
			abbr: l,
			acronym: l,
			address: l,
			applet: l,
			area: {
				attrs: {
					alt: null,
					coords: null,
					href: null,
					target: null,
					ping: null,
					media: s,
					hreflang: n,
					type: null,
					shape: ["default", "rect", "circle", "poly"]
				}
			},
			article: l,
			aside: l,
			audio: {
				attrs: {
					src: null,
					mediagroup: null,
					crossorigin: ["anonymous", "use-credentials"],
					preload: ["none", "metadata", "auto"],
					autoplay: ["", "autoplay"],
					loop: ["", "loop"],
					controls: ["", "controls"]
				}
			},
			b: l,
			base: {
				attrs: {
					href: null,
					target: i
				}
			},
			basefont: l,
			bdi: l,
			bdo: l,
			big: l,
			blockquote: {
				attrs: {
					cite: null
				}
			},
			body: l,
			br: l,
			button: {
				attrs: {
					form: null,
					formaction: null,
					name: null,
					value: null,
					autofocus: ["", "autofocus"],
					disabled: ["", "autofocus"],
					formenctype: a,
					formmethod: o,
					formnovalidate: ["", "novalidate"],
					formtarget: i,
					type: ["submit", "reset", "button"]
				}
			},
			canvas: {
				attrs: {
					width: null,
					height: null
				}
			},
			caption: l,
			center: l,
			cite: l,
			code: l,
			col: {
				attrs: {
					span: null
				}
			},
			colgroup: {
				attrs: {
					span: null
				}
			},
			command: {
				attrs: {
					type: ["command", "checkbox", "radio"],
					label: null,
					icon: null,
					radiogroup: null,
					command: null,
					title: null,
					disabled: ["", "disabled"],
					checked: ["", "checked"]
				}
			},
			data: {
				attrs: {
					value: null
				}
			},
			datagrid: {
				attrs: {
					disabled: ["", "disabled"],
					multiple: ["", "multiple"]
				}
			},
			datalist: {
				attrs: {
					data: null
				}
			},
			dd: l,
			del: {
				attrs: {
					cite: null,
					datetime: null
				}
			},
			details: {
				attrs: {
					open: ["", "open"]
				}
			},
			dfn: l,
			dir: l,
			div: l,
			dl: l,
			dt: l,
			em: l,
			embed: {
				attrs: {
					src: null,
					type: null,
					width: null,
					height: null
				}
			},
			eventsource: {
				attrs: {
					src: null
				}
			},
			fieldset: {
				attrs: {
					disabled: ["", "disabled"],
					form: null,
					name: null
				}
			},
			figcaption: l,
			figure: l,
			font: l,
			footer: l,
			form: {
				attrs: {
					action: null,
					name: null,
					"accept-charset": r,
					autocomplete: ["on", "off"],
					enctype: a,
					method: o,
					novalidate: ["", "novalidate"],
					target: i
				}
			},
			frame: l,
			frameset: l,
			h1: l,
			h2: l,
			h3: l,
			h4: l,
			h5: l,
			h6: l,
			head: {
				attrs: {},
				children: ["title", "base", "link", "style", "meta", "script", "noscript", "command"]
			},
			header: l,
			hgroup: l,
			hr: l,
			html: {
				attrs: {
					manifest: null
				},
				children: ["head", "body"]
			},
			i: l,
			iframe: {
				attrs: {
					src: null,
					srcdoc: null,
					name: null,
					width: null,
					height: null,
					sandbox: ["allow-top-navigation", "allow-same-origin", "allow-forms", "allow-scripts"],
					seamless: ["", "seamless"]
				}
			},
			img: {
				attrs: {
					alt: null,
					src: null,
					ismap: null,
					usemap: null,
					width: null,
					height: null,
					crossorigin: ["anonymous", "use-credentials"]
				}
			},
			input: {
				attrs: {
					alt: null,
					dirname: null,
					form: null,
					formaction: null,
					height: null,
					list: null,
					max: null,
					maxlength: null,
					min: null,
					name: null,
					pattern: null,
					placeholder: null,
					size: null,
					src: null,
					step: null,
					value: null,
					width: null,
					accept: ["audio/*", "video/*", "image/*"],
					autocomplete: ["on", "off"],
					autofocus: ["", "autofocus"],
					checked: ["", "checked"],
					disabled: ["", "disabled"],
					formenctype: a,
					formmethod: o,
					formnovalidate: ["", "novalidate"],
					formtarget: i,
					multiple: ["", "multiple"],
					readonly: ["", "readonly"],
					required: ["", "required"],
					type: ["hidden", "text", "search", "tel", "url", "email", "password", "datetime", "date", "month", "week", "time", "datetime-local", "number", "range", "color", "checkbox", "radio", "file", "submit", "image", "reset", "button"]
				}
			},
			ins: {
				attrs: {
					cite: null,
					datetime: null
				}
			},
			kbd: l,
			keygen: {
				attrs: {
					challenge: null,
					form: null,
					name: null,
					autofocus: ["", "autofocus"],
					disabled: ["", "disabled"],
					keytype: ["RSA"]
				}
			},
			label: {
				attrs: {
					for: null,
					form: null
				}
			},
			legend: l,
			li: {
				attrs: {
					value: null
				}
			},
			link: {
				attrs: {
					href: null,
					type: null,
					hreflang: n,
					media: s,
					sizes: ["all", "16x16", "16x16 32x32", "16x16 32x32 64x64"]
				}
			},
			map: {
				attrs: {
					name: null
				}
			},
			mark: l,
			menu: {
				attrs: {
					label: null,
					type: ["list", "context", "toolbar"]
				}
			},
			meta: {
				attrs: {
					content: null,
					charset: r,
					name: ["viewport", "application-name", "author", "description", "generator", "keywords"],
					"http-equiv": ["content-language", "content-type", "default-style", "refresh"]
				}
			},
			meter: {
				attrs: {
					value: null,
					min: null,
					low: null,
					high: null,
					max: null,
					optimum: null
				}
			},
			nav: l,
			noframes: l,
			noscript: l,
			object: {
				attrs: {
					data: null,
					type: null,
					name: null,
					usemap: null,
					form: null,
					width: null,
					height: null,
					typemustmatch: ["", "typemustmatch"]
				}
			},
			ol: {
				attrs: {
					reversed: ["", "reversed"],
					start: null,
					type: ["1", "a", "A", "i", "I"]
				}
			},
			optgroup: {
				attrs: {
					disabled: ["", "disabled"],
					label: null
				}
			},
			option: {
				attrs: {
					disabled: ["", "disabled"],
					label: null,
					selected: ["", "selected"],
					value: null
				}
			},
			output: {
				attrs: {
					for: null,
					form: null,
					name: null
				}
			},
			p: l,
			param: {
				attrs: {
					name: null,
					value: null
				}
			},
			pre: l,
			progress: {
				attrs: {
					value: null,
					max: null
				}
			},
			q: {
				attrs: {
					cite: null
				}
			},
			rp: l,
			rt: l,
			ruby: l,
			s: l,
			samp: l,
			script: {
				attrs: {
					type: ["text/javascript"],
					src: null,
					async: ["", "async"],
					defer: ["", "defer"],
					charset: r
				}
			},
			section: l,
			select: {
				attrs: {
					form: null,
					name: null,
					size: null,
					autofocus: ["", "autofocus"],
					disabled: ["", "disabled"],
					multiple: ["", "multiple"]
				}
			},
			small: l,
			source: {
				attrs: {
					src: null,
					type: null,
					media: null
				}
			},
			span: l,
			strike: l,
			strong: l,
			style: {
				attrs: {
					type: ["text/css"],
					media: s,
					scoped: null
				}
			},
			sub: l,
			summary: l,
			sup: l,
			table: l,
			tbody: l,
			td: {
				attrs: {
					colspan: null,
					rowspan: null,
					headers: null
				}
			},
			textarea: {
				attrs: {
					dirname: null,
					form: null,
					maxlength: null,
					name: null,
					placeholder: null,
					rows: null,
					cols: null,
					autofocus: ["", "autofocus"],
					disabled: ["", "disabled"],
					readonly: ["", "readonly"],
					required: ["", "required"],
					wrap: ["soft", "hard"]
				}
			},
			tfoot: l,
			th: {
				attrs: {
					colspan: null,
					rowspan: null,
					headers: null,
					scope: ["row", "col", "rowgroup", "colgroup"]
				}
			},
			thead: l,
			time: {
				attrs: {
					datetime: null
				}
			},
			title: l,
			tr: l,
			track: {
				attrs: {
					src: null,
					label: null,
				default:
					null,
					kind: ["subtitles", "captions", "descriptions", "chapters", "metadata"],
					srclang: n
				}
			},
			tt: l,
			u: l,
			ul: l,
			var: l,
			video: {
				attrs: {
					src: null,
					poster: null,
					width: null,
					height: null,
					crossorigin: ["anonymous", "use-credentials"],
					preload: ["auto", "metadata", "none"],
					autoplay: ["", "autoplay"],
					mediagroup: ["movie"],
					muted: ["", "muted"],
					controls: ["", "controls"]
				}
			},
			wbr: l
		},
		u = {
			accesskey: ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
			class: null,
			contenteditable: ["true", "false"],
			contextmenu: null,
			dir: ["ltr", "rtl", "auto"],
			draggable: ["true", "false", "auto"],
			dropzone: ["copy", "move", "link", "string:", "file:"],
			hidden: ["hidden"],
			id: null,
			inert: ["inert"],
			itemid: null,
			itemprop: null,
			itemref: null,
			itemscope: ["itemscope"],
			itemtype: null,
			lang: ["en", "es"],
			spellcheck: ["true", "false"],
			style: null,
			tabindex: ["1", "2", "3", "4", "5", "6", "7", "8", "9"],
			title: null,
			translate: ["yes", "no"],
			onclick: null,
			rel: ["stylesheet", "alternate", "author", "bookmark", "help", "license", "next", "nofollow", "noreferrer", "prefetch", "prev", "search", "tag"]
		};
		t(l);
		for (var d in c) c.hasOwnProperty(d) && c[d] != l && t(c[d]);
		e.htmlSchema = c,
		e.registerHelper("hint", "html",
		function(t, n) {
			var i = {
				schemaInfo: c
			};
			if (n) for (var r in n) i[r] = n[r];
			return e.hint.xml(t, i)
		})
	}),
	function(e) {
		"object" == typeof exports && "object" == typeof module ? e(require("../../lib/codemirror")) : "function" == typeof define && define.amd ? define(["../../lib/codemirror"], e) : e(CodeMirror)
	} (function(e) {
		function t(e, t) {
			for (var n = 0,
			i = e.length; i > n; ++n) t(e[n])
		}
		function n(e, t) {
			if (!Array.prototype.indexOf) {
				for (var n = e.length; n--;) if (e[n] === t) return ! 0;
				return ! 1
			}
			return - 1 != e.indexOf(t)
		}
		function i(t, n, i, r) {
			var s = t.getCursor(),
			l = i(t, s);
			if (!/\b(?:string|comment)\b/.test(l.type)) {
				l.state = e.innerMode(t.getMode(), l.state).state,
				/^[\w$_]*$/.test(l.string) ? l.end > s.ch && (l.end = s.ch, l.string = l.string.slice(0, s.ch - l.start)) : l = {
					start: s.ch,
					end: s.ch,
					string: "",
					state: l.state,
					type: "." == l.string ? "property": null
				};
				for (var c = l;
				"property" == c.type;) {
					if ("." != (c = i(t, a(s.line, c.start))).string) return;
					if (c = i(t, a(s.line, c.start)), !u) var u = [];
					u.push(c)
				}
				return {
					list: o(l, u, n, r),
					from: a(s.line, l.start),
					to: a(s.line, l.end)
				}
			}
		}
		function r(e, t) {
			var n = e.getTokenAt(t);
			return t.ch == n.start + 1 && "." == n.string.charAt(0) ? (n.end = n.start, n.string = ".", n.type = "property") : /^\.[\w$_]*$/.test(n.string) && (n.type = "property", n.start++, n.string = n.string.replace(/\./, "")),
			n
		}
		function o(e, i, r, o) {
			function a(e) {
				0 != e.lastIndexOf(h, 0) || n(d, e) || d.push(e)
			}
			function u(e) {
				"string" == typeof e ? t(s, a) : e instanceof Array ? t(l, a) : e instanceof Function && t(c, a);
				for (var n in e) a(n)
			}
			var d = [],
			h = e.string,
			f = o && o.globalScope || window;
			if (i && i.length) {
				var p, m = i.pop();
				for (m.type && 0 === m.type.indexOf("variable") ? (o && o.additionalContext && (p = o.additionalContext[m.string]), o && !1 === o.useGlobalScope || (p = p || f[m.string])) : "string" == m.type ? p = "": "atom" == m.type ? p = 1 : "function" == m.type && (null == f.jQuery || "$" != m.string && "jQuery" != m.string || "function" != typeof f.jQuery ? null != f._ && "_" == m.string && "function" == typeof f._ && (p = f._()) : p = f.jQuery()); null != p && i.length;) p = p[i.pop().string];
				null != p && u(p)
			} else {
				for (g = e.state.localVars; g; g = g.next) a(g.name);
				for (var g = e.state.globalVars; g; g = g.next) a(g.name);
				o && !1 === o.useGlobalScope || u(f),
				t(r, a)
			}
			return d
		}
		var a = e.Pos;
		e.registerHelper("hint", "javascript",
		function(e, t) {
			return i(e, u,
			function(e, t) {
				return e.getTokenAt(t)
			},
			t)
		}),
		e.registerHelper("hint", "coffeescript",
		function(e, t) {
			return i(e, d, r, t)
		});
		var s = "charAt charCodeAt indexOf lastIndexOf substring substr slice trim trimLeft trimRight toUpperCase toLowerCase split concat match replace search".split(" "),
		l = "length concat join splice push pop shift unshift slice reverse sort indexOf lastIndexOf every some filter forEach map reduce reduceRight ".split(" "),
		c = "prototype apply call bind".split(" "),
		u = "break case catch continue debugger default delete do else false finally for function if in instanceof new null return switch throw true try typeof var void while with".split(" "),
		d = "and break catch class continue delete do else extends false finally for if in instanceof isnt new no not null of off on or return switch then throw true try typeof until void while with yes".split(" ")
	}),
	function(e) {
		"object" == typeof exports && "object" == typeof module ? e(require("../../lib/codemirror")) : "function" == typeof define && define.amd ? define(["../../lib/codemirror"], e) : e(CodeMirror)
	} (function(e) {
		"use strict";
		function t(e) {
			e.operation(function() {
				a(e)
			})
		}
		function n(e) {
			e.state.markedSelection.length && e.operation(function() {
				r(e)
			})
		}
		function i(e, t, n, i) {
			if (0 != c(t, n)) for (var r = e.state.markedSelection,
			o = e.state.markedSelectionStyle,
			a = t.line;;) {
				var u = a == t.line ? t: l(a, 0),
				d = a + s,
				h = d >= n.line,
				f = h ? n: l(d, 0),
				p = e.markText(u, f, {
					className: o
				});
				if (null == i ? r.push(p) : r.splice(i++, 0, p), h) break;
				a = d
			}
		}
		function r(e) {
			for (var t = e.state.markedSelection,
			n = 0; n < t.length; ++n) t[n].clear();
			t.length = 0
		}
		function o(e) {
			r(e);
			for (var t = e.listSelections(), n = 0; n < t.length; n++) i(e, t[n].from(), t[n].to())
		}
		function a(e) {
			if (!e.somethingSelected()) return r(e);
			if (e.listSelections().length > 1) return o(e);
			var t = e.getCursor("start"),
			n = e.getCursor("end"),
			a = e.state.markedSelection;
			if (!a.length) return i(e, t, n);
			var l = a[0].find(),
			u = a[a.length - 1].find();
			if (!l || !u || n.line - t.line < s || c(t, u.to) >= 0 || c(n, l.from) <= 0) return o(e);
			for (; c(t, l.from) > 0;) a.shift().clear(),
			l = a[0].find();
			for (c(t, l.from) < 0 && (l.to.line - t.line < s ? (a.shift().clear(), i(e, t, l.to, 0)) : i(e, t, l.from, 0)); c(n, u.to) < 0;) a.pop().clear(),
			u = a[a.length - 1].find();
			c(n, u.to) > 0 && (n.line - u.from.line < s ? (a.pop().clear(), i(e, u.from, n)) : i(e, u.to, n))
		}
		e.defineOption("styleSelectedText", !1,
		function(i, a, s) {
			var l = s && s != e.Init;
			a && !l ? (i.state.markedSelection = [], i.state.markedSelectionStyle = "string" == typeof a ? a: "CodeMirror-selectedtext", o(i), i.on("cursorActivity", t), i.on("change", n)) : !a && l && (i.off("cursorActivity", t), i.off("change", n), r(i), i.state.markedSelection = i.state.markedSelectionStyle = null)
		});
		var s = 8,
		l = e.Pos,
		c = e.cmpPos
	}),
	function(e) {
		"object" == typeof exports && "object" == typeof module ? e(require("../../lib/codemirror"), require("./matchesonscrollbar")) : "function" == typeof define && define.amd ? define(["../../lib/codemirror", "./matchesonscrollbar"], e) : e(CodeMirror)
	} (function(e) {
		"use strict";
		function t(e) {
			"object" == typeof e && (this.minChars = e.minChars, this.style = e.style, this.showToken = e.showToken, this.delay = e.delay, this.wordsOnly = e.wordsOnly, this.annotateScrollbar = e.annotateScrollbar),
			null == this.style && (this.style = u),
			null == this.minChars && (this.minChars = c),
			null == this.delay && (this.delay = d),
			null == this.wordsOnly && (this.wordsOnly = h),
			this.overlay = this.timeout = null,
			this.matchesonscroll = null
		}
		function n(e) {
			var t = e.state.matchHighlighter;
			clearTimeout(t.timeout),
			t.timeout = setTimeout(function() {
				o(e)
			},
			t.delay)
		}
		function i(e, t, n, i) {
			var r = e.state.matchHighlighter;
			if (e.addOverlay(r.overlay = l(t, n, i)), r.annotateScrollbar) {
				var o = n ? new RegExp("\\b" + t + "\\b") : t;
				r.matchesonscroll = e.showMatchesOnScrollbar(o, !0, {
					className: "CodeMirror-selection-highlight-scrollbar"
				})
			}
		}
		function r(e) {
			var t = e.state.matchHighlighter;
			t.overlay && (e.removeOverlay(t.overlay), t.overlay = null, t.annotateScrollbar && (t.matchesonscroll.clear(), t.matchesonscroll = null))
		}
		function o(e) {
			e.operation(function() {
				var t = e.state.matchHighlighter;
				if (r(e), e.somethingSelected() || !t.showToken) {
					var n = e.getCursor("from"),
					o = e.getCursor("to");
					if (n.line == o.line && (!t.wordsOnly || a(e, n, o))) {
						var s = e.getRange(n, o).replace(/^\s+|\s+$/g, "");
						s.length >= t.minChars && i(e, s, !1, t.style)
					}
				} else {
					for (var l = !0 === t.showToken ? /[\w$]/: t.showToken, c = e.getCursor(), u = e.getLine(c.line), d = c.ch, h = d; d && l.test(u.charAt(d - 1));)--d;
					for (; h < u.length && l.test(u.charAt(h));)++h;
					h > d && i(e, u.slice(d, h), l, t.style)
				}
			})
		}
		function a(e, t, n) {
			if (null !== e.getRange(t, n).match(/^\w+$/)) {
				if (t.ch > 0) {
					i = {
						line: t.line,
						ch: t.ch - 1
					};
					if (null === (r = e.getRange(i, t)).match(/\W/)) return ! 1
				}
				if (n.ch < e.getLine(t.line).length) {
					var i = {
						line: n.line,
						ch: n.ch + 1
					},
					r = e.getRange(n, i);
					if (null === r.match(/\W/)) return ! 1
				}
				return ! 0
			}
			return ! 1
		}
		function s(e, t) {
			return ! (e.start && t.test(e.string.charAt(e.start - 1)) || e.pos != e.string.length && t.test(e.string.charAt(e.pos)))
		}
		function l(e, t, n) {
			return {
				token: function(i) {
					return ! i.match(e) || t && !s(i, t) ? (i.next(), void(i.skipTo(e.charAt(0)) || i.skipToEnd())) : n
				}
			}
		}
		var c = 2,
		u = "matchhighlight",
		d = 100,
		h = !1;
		e.defineOption("highlightSelectionMatches", !1,
		function(i, a, s) {
			s && s != e.Init && (r(i), clearTimeout(i.state.matchHighlighter.timeout), i.state.matchHighlighter = null, i.off("cursorActivity", n)),
			a && (i.state.matchHighlighter = new t(a), o(i), i.on("cursorActivity", n))
		})
	}),
	function(e) {
		"object" == typeof exports && "object" == typeof module ? e(require("../../lib/codemirror")) : "function" == typeof define && define.amd ? define(["../../lib/codemirror"], e) : e(CodeMirror)
	} (function(e) {
		function t(e, t, i, r) {
			var o = e.getLineHandle(t.line),
			l = t.ch - 1,
			c = l >= 0 && s[o.text.charAt(l)] || s[o.text.charAt(++l)];
			if (!c) return null;
			var u = ">" == c.charAt(1) ? 1 : -1;
			if (i && u > 0 != (l == t.ch)) return null;
			var d = e.getTokenTypeAt(a(t.line, l + 1)),
			h = n(e, a(t.line, l + (u > 0 ? 1 : 0)), u, d || null, r);
			return null == h ? null: {
				from: a(t.line, l),
				to: h && h.pos,
				match: h && h.ch == c.charAt(0),
				forward: u > 0
			}
		}
		function n(e, t, n, i, r) {
			for (var o = r && r.maxScanLineLength || 1e4,
			l = r && r.maxScanLines || 1e3,
			c = [], u = r && r.bracketRegex ? r.bracketRegex: /[(){}[\]]/, d = n > 0 ? Math.min(t.line + l, e.lastLine() + 1) : Math.max(e.firstLine() - 1, t.line - l), h = t.line; h != d; h += n) {
				var f = e.getLine(h);
				if (f) {
					var p = n > 0 ? 0 : f.length - 1,
					m = n > 0 ? f.length: -1;
					if (! (f.length > o)) for (h == t.line && (p = t.ch - (0 > n ? 1 : 0)); p != m; p += n) {
						var g = f.charAt(p);
						if (u.test(g) && (void 0 === i || e.getTokenTypeAt(a(h, p + 1)) == i)) if (">" == s[g].charAt(1) == n > 0) c.push(g);
						else {
							if (!c.length) return {
								pos: a(h, p),
								ch: g
							};
							c.pop()
						}
					}
				}
			}
			return h - n != (n > 0 ? e.lastLine() : e.firstLine()) && null
		}
		function i(e, n, i) {
			for (var r = e.state.matchBrackets.maxHighlightLineLength || 1e3,
			s = [], l = e.listSelections(), c = 0; c < l.length; c++) {
				var u = l[c].empty() && t(e, l[c].head, !1, i);
				if (u && e.getLine(u.from.line).length <= r) {
					var d = u.match ? "CodeMirror-matchingbracket": "CodeMirror-nonmatchingbracket";
					s.push(e.markText(u.from, a(u.from.line, u.from.ch + 1), {
						className: d
					})),
					u.to && e.getLine(u.to.line).length <= r && s.push(e.markText(u.to, a(u.to.line, u.to.ch + 1), {
						className: d
					}))
				}
			}
			if (s.length) {
				o && e.state.focused && e.focus();
				var h = function() {
					e.operation(function() {
						for (var e = 0; e < s.length; e++) s[e].clear()
					})
				};
				if (!n) return h;
				setTimeout(h, 800)
			}
		}
		function r(e) {
			e.operation(function() {
				l && (l(), l = null),
				l = i(e, !1, e.state.matchBrackets)
			})
		}
		var o = /MSIE \d/.test(navigator.userAgent) && (null == document.documentMode || document.documentMode < 8),
		a = e.Pos,
		s = {
			"(": ")>",
			")": "(<",
			"[": "]>",
			"]": "[<",
			"{": "}>",
			"}": "{<"
		},
		l = null;
		e.defineOption("matchBrackets", !1,
		function(t, n, i) {
			i && i != e.Init && t.off("cursorActivity", r),
			n && (t.state.matchBrackets = "object" == typeof n ? n: {},
			t.on("cursorActivity", r))
		}),
		e.defineExtension("matchBrackets",
		function() {
			i(this, !0)
		}),
		e.defineExtension("findMatchingBracket",
		function(e, n, i) {
			return t(this, e, n, i)
		}),
		e.defineExtension("scanForBracket",
		function(e, t, i, r) {
			return n(this, e, t, i, r)
		})
	}),
	function(e) {
		"object" == typeof exports && "object" == typeof module ? e(require("../../lib/codemirror"), require("../fold/xml-fold")) : "function" == typeof define && define.amd ? define(["../../lib/codemirror", "../fold/xml-fold"], e) : e(CodeMirror)
	} (function(e) {
		"use strict";
		function t(e) {
			e.state.tagHit && e.state.tagHit.clear(),
			e.state.tagOther && e.state.tagOther.clear(),
			e.state.tagHit = e.state.tagOther = null
		}
		function n(n) {
			n.state.failedTagMatch = !1,
			n.operation(function() {
				if (t(n), !n.somethingSelected()) {
					var i = n.getCursor(),
					r = n.getViewport();
					r.from = Math.min(r.from, i.line),
					r.to = Math.max(i.line + 1, r.to);
					var o = e.findMatchingTag(n, i, r);
					if (o) {
						if (n.state.matchBothTags) {
							var a = "open" == o.at ? o.open: o.close;
							a && (n.state.tagHit = n.markText(a.from, a.to, {
								className: "CodeMirror-matchingtag"
							}))
						}
						var s = "close" == o.at ? o.open: o.close;
						s ? n.state.tagOther = n.markText(s.from, s.to, {
							className: "CodeMirror-matchingtag"
						}) : n.state.failedTagMatch = !0
					}
				}
			})
		}
		function i(e) {
			e.state.failedTagMatch && n(e)
		}
		e.defineOption("matchTags", !1,
		function(r, o, a) {
			a && a != e.Init && (r.off("cursorActivity", n), r.off("viewportChange", i), t(r)),
			o && (r.state.matchBothTags = "object" == typeof o && o.bothTags, r.on("cursorActivity", n), r.on("viewportChange", i), n(r))
		}),
		e.commands.toMatchingTag = function(t) {
			var n = e.findMatchingTag(t, t.getCursor());
			if (n) {
				var i = "close" == n.at ? n.open: n.close;
				i && t.extendSelection(i.to, i.from)
			}
		}
	}),
	function(e) {
		"object" == typeof exports && "object" == typeof module ? e(require("../../lib/codemirror")) : "function" == typeof define && define.amd ? define(["../../lib/codemirror"], e) : e(CodeMirror)
	} (function(e) {
		"use strict";
		e.overlayMode = function(t, n, i) {
			return {
				startState: function() {
					return {
						base: e.startState(t),
						overlay: e.startState(n),
						basePos: 0,
						baseCur: null,
						overlayPos: 0,
						overlayCur: null,
						streamSeen: null
					}
				},
				copyState: function(i) {
					return {
						base: e.copyState(t, i.base),
						overlay: e.copyState(n, i.overlay),
						basePos: i.basePos,
						baseCur: null,
						overlayPos: i.overlayPos,
						overlayCur: null
					}
				},
				token: function(e, r) {
					return (e != r.streamSeen || Math.min(r.basePos, r.overlayPos) < e.start) && (r.streamSeen = e, r.basePos = r.overlayPos = e.start),
					e.start == r.basePos && (r.baseCur = t.token(e, r.base), r.basePos = e.pos),
					e.start == r.overlayPos && (e.pos = e.start, r.overlayCur = n.token(e, r.overlay), r.overlayPos = e.pos),
					e.pos = Math.min(r.basePos, r.overlayPos),
					null == r.overlayCur ? r.baseCur: null != r.baseCur && r.overlay.combineTokens || i && null == r.overlay.combineTokens ? r.baseCur + " " + r.overlayCur: r.overlayCur
				},
				indent: t.indent &&
				function(e, n) {
					return t.indent(e.base, n)
				},
				electricChars: t.electricChars,
				innerMode: function(e) {
					return {
						state: e.base,
						mode: t
					}
				},
				blankLine: function(e) {
					t.blankLine && t.blankLine(e.base),
					n.blankLine && n.blankLine(e.overlay)
				}
			}
		}
	}),
	function(e) {
		"object" == typeof exports && "object" == typeof module ? e(require("../../lib/codemirror")) : "function" == typeof define && define.amd ? define(["../../lib/codemirror"], e) : e(CodeMirror)
	} (function(e) {
		function t(e) {
			e.state.placeholder && (e.state.placeholder.parentNode.removeChild(e.state.placeholder), e.state.placeholder = null)
		}
		function n(e) {
			t(e);
			var n = e.state.placeholder = document.createElement("pre");
			n.style.cssText = "height: 0; overflow: visible",
			n.className = "CodeMirror-placeholder";
			var i = e.getOption("placeholder");
			"string" == typeof i && (i = document.createTextNode(i)),
			n.appendChild(i),
			e.display.lineSpace.insertBefore(n, e.display.lineSpace.firstChild)
		}
		function i(e) {
			o(e) && n(e)
		}
		function r(e) {
			var i = e.getWrapperElement(),
			r = o(e);
			i.className = i.className.replace(" CodeMirror-empty", "") + (r ? " CodeMirror-empty": ""),
			r ? n(e) : t(e)
		}
		function o(e) {
			return 1 === e.lineCount() && "" === e.getLine(0)
		}
		e.defineOption("placeholder", "",
		function(n, o, a) {
			var s = a && a != e.Init;
			if (o && !s) n.on("blur", i),
			n.on("change", r),
			n.on("swapDoc", r),
			r(n);
			else if (!o && s) {
				n.off("blur", i),
				n.off("change", r),
				n.off("swapDoc", r),
				t(n);
				var l = n.getWrapperElement();
				l.className = l.className.replace(" CodeMirror-empty", "")
			}
			o && !n.hasFocus() && i(n)
		})
	}),
	function(e) {
		"object" == typeof exports && "object" == typeof module ? e(require("../../lib/codemirror")) : "function" == typeof define && define.amd ? define(["../../lib/codemirror"], e) : e(CodeMirror)
	} (function(e) {
		"use strict";
		function t(e, t) {
			this.cm = e,
			this.options = t,
			this.widget = null,
			this.debounce = 0,
			this.tick = 0,
			this.startPos = this.cm.getCursor("start"),
			this.startLen = this.cm.getLine(this.startPos.line).length - this.cm.getSelection().length;
			var n = this;
			e.on("cursorActivity", this.activityFunc = function() {
				n.cursorActivity()
			})
		}
		function n(t, n) {
			return e.cmpPos(n.from, t.from) > 0 && t.to.ch - t.from.ch != n.to.ch - n.from.ch
		}
		function i(e, t, n) {
			var i = e.options.hintOptions,
			r = {};
			for (var o in f) r[o] = f[o];
			if (i) for (var o in i) void 0 !== i[o] && (r[o] = i[o]);
			if (n) for (var o in n) void 0 !== n[o] && (r[o] = n[o]);
			return r.hint.resolve && (r.hint = r.hint.resolve(e, t)),
			r
		}
		function r(e) {
			return "string" == typeof e ? e: e.text
		}
		function o(e, t) {
			function n(e, n) {
				var r;
				r = "string" != typeof n ?
				function(e) {
					return n(e, t)
				}: i.hasOwnProperty(n) ? i[n] : n,
				o[e] = r
			}
			var i = {
				Up: function() {
					t.moveFocus( - 1)
				},
				Down: function() {
					t.moveFocus(1)
				},
				PageUp: function() {
					t.moveFocus(1 - t.menuSize(), !0)
				},
				PageDown: function() {
					t.moveFocus(t.menuSize() - 1, !0)
				},
				Home: function() {
					t.setFocus(0)
				},
				End: function() {
					t.setFocus(t.length - 1)
				},
				Enter: t.pick,
				Tab: t.pick,
				Esc: t.close
			},
			r = e.options.customKeys,
			o = r ? {}: i;
			if (r) for (var a in r) r.hasOwnProperty(a) && n(a, r[a]);
			var s = e.options.extraKeys;
			if (s) for (var a in s) s.hasOwnProperty(a) && n(a, s[a]);
			return o
		}
		function a(e, t) {
			for (; t && t != e;) {
				if ("LI" === t.nodeName.toUpperCase() && t.parentNode == e) return t;
				t = t.parentNode
			}
		}
		function s(t, n) {
			this.completion = t,
			this.data = n,
			this.picked = !1;
			var i = this,
			s = t.cm,
			l = this.hints = document.createElement("ul");
			l.className = "CodeMirror-hints",
			this.selectedHint = n.selectedHint || 0;
			for (var d = n.list,
			h = 0; h < d.length; ++h) {
				var f = l.appendChild(document.createElement("li")),
				p = d[h],
				m = c + (h != this.selectedHint ? "": " " + u);
				null != p.className && (m = p.className + " " + m),
				f.className = m,
				p.render ? p.render(f, n, p) : f.appendChild(document.createTextNode(p.displayText || r(p))),
				f.hintId = h
			}
			var g = s.cursorCoords(t.options.alignWithWord ? n.from: null),
			v = g.left,
			y = g.bottom,
			b = !0;
			l.style.left = v + "px",
			l.style.top = y + "px";
			var w = window.innerWidth || Math.max(document.body.offsetWidth, document.documentElement.offsetWidth),
			k = window.innerHeight || Math.max(document.body.offsetHeight, document.documentElement.offsetHeight); (t.options.container || document.body).appendChild(l);
			var x = l.getBoundingClientRect();
			if (x.bottom - k > 0) {
				var _ = x.bottom - x.top;
				if (g.top - (g.bottom - x.top) - _ > 0) l.style.top = (y = g.top - _) + "px",
				b = !1;
				else if (_ > k) {
					l.style.height = k - 5 + "px",
					l.style.top = (y = g.bottom - x.top) + "px";
					var C = s.getCursor();
					n.from.ch != C.ch && (g = s.cursorCoords(C), l.style.left = (v = g.left) + "px", x = l.getBoundingClientRect())
				}
			}
			var S = x.right - w;
			if (S > 0 && (x.right - x.left > w && (l.style.width = w - 5 + "px", S -= x.right - x.left - w), l.style.left = (v = g.left - S) + "px"), s.addKeyMap(this.keyMap = o(t, {
				moveFocus: function(e, t) {
					i.changeActive(i.selectedHint + e, t)
				},
				setFocus: function(e) {
					i.changeActive(e)
				},
				menuSize: function() {
					return i.screenAmount()
				},
				length: d.length,
				close: function() {
					t.close()
				},
				pick: function() {
					i.pick()
				},
				data: n
			})), t.options.closeOnUnfocus) {
				var M;
				s.on("blur", this.onBlur = function() {
					M = setTimeout(function() {
						t.close()
					},
					100)
				}),
				s.on("focus", this.onFocus = function() {
					clearTimeout(M)
				})
			}
			var T = s.getScrollInfo();
			return s.on("scroll", this.onScroll = function() {
				var e = s.getScrollInfo(),
				n = s.getWrapperElement().getBoundingClientRect(),
				i = y + T.top - e.top,
				r = i - (window.pageYOffset || (document.documentElement || document.body).scrollTop);
				return b || (r += l.offsetHeight),
				r <= n.top || r >= n.bottom ? t.close() : (l.style.top = i + "px", void(l.style.left = v + T.left - e.left + "px"))
			}),
			e.on(l, "dblclick",
			function(e) {
				var t = a(l, e.target || e.srcElement);
				t && null != t.hintId && (i.changeActive(t.hintId), i.pick())
			}),
			e.on(l, "click",
			function(e) {
				var n = a(l, e.target || e.srcElement);
				n && null != n.hintId && (i.changeActive(n.hintId), t.options.completeOnSingleClick && i.pick())
			}),
			e.on(l, "mousedown",
			function() {
				setTimeout(function() {
					s.focus()
				},
				20)
			}),
			e.signal(n, "select", d[0], l.firstChild),
			!0
		}
		function l(e, t) {
			if (!e.somethingSelected()) return t;
			for (var n = [], i = 0; i < t.length; i++) t[i].supportsSelection && n.push(t[i]);
			return n
		}
		var c = "CodeMirror-hint",
		u = "CodeMirror-hint-active";
		e.showHint = function(e, t, n) {
			if (!t) return e.showHint(n);
			n && n.async && (t.async = !0);
			var i = {
				hint: t
			};
			if (n) for (var r in n) i[r] = n[r];
			return e.showHint(i)
		},
		e.defineExtension("showHint",
		function(n) {
			n = i(this, this.getCursor("start"), n);
			var r = this.listSelections();
			if (! (r.length > 1)) {
				if (this.somethingSelected()) {
					if (!n.hint.supportsSelection) return;
					for (var o = 0; o < r.length; o++) if (r[o].head.line != r[o].anchor.line) return
				}
				this.state.completionActive && this.state.completionActive.close();
				var a = this.state.completionActive = new t(this, n);
				a.options.hint && (e.signal(this, "startCompletion", this), a.update(!0))
			}
		});
		var d = window.requestAnimationFrame ||
		function(e) {
			return setTimeout(e, 1e3 / 60)
		},
		h = window.cancelAnimationFrame || clearTimeout;
		t.prototype = {
			close: function() {
				this.active() && (this.cm.state.completionActive = null, this.tick = null, this.cm.off("cursorActivity", this.activityFunc), this.widget && this.data && e.signal(this.data, "close"), this.widget && this.widget.close(), e.signal(this.cm, "endCompletion", this.cm))
			},
			active: function() {
				return this.cm.state.completionActive == this
			},
			pick: function(t, n) {
				var i = t.list[n];
				i.hint ? i.hint(this.cm, t, i) : this.cm.replaceRange(r(i), i.from || t.from, i.to || t.to, "complete"),
				e.signal(t, "pick", i),
				this.close()
			},
			cursorActivity: function() {
				this.debounce && (h(this.debounce), this.debounce = 0);
				var e = this.cm.getCursor(),
				t = this.cm.getLine(e.line);
				if (e.line != this.startPos.line || t.length - e.ch != this.startLen - this.startPos.ch || e.ch < this.startPos.ch || this.cm.somethingSelected() || e.ch && this.options.closeCharacters.test(t.charAt(e.ch - 1))) this.close();
				else {
					var n = this;
					this.debounce = d(function() {
						n.update()
					}),
					this.widget && this.widget.disable()
				}
			},
			update: function(e) {
				if (null != this.tick) if (this.options.hint.async) {
					var t = ++this.tick,
					n = this;
					this.options.hint(this.cm,
					function(i) {
						n.tick == t && n.finishUpdate(i, e)
					},
					this.options)
				} else this.finishUpdate(this.options.hint(this.cm, this.options), e)
			},
			finishUpdate: function(t, i) {
				this.data && e.signal(this.data, "update");
				var r = this.widget && this.widget.picked || i && this.options.completeSingle;
				this.widget && this.widget.close(),
				t && this.data && n(this.data, t) || (this.data = t, t && t.list.length && (r && 1 == t.list.length ? this.pick(t, 0) : (this.widget = new s(this, t), e.signal(t, "shown"))))
			}
		},
		s.prototype = {
			close: function() {
				if (this.completion.widget == this) {
					this.completion.widget = null,
					this.hints.parentNode.removeChild(this.hints),
					this.completion.cm.removeKeyMap(this.keyMap);
					var e = this.completion.cm;
					this.completion.options.closeOnUnfocus && (e.off("blur", this.onBlur), e.off("focus", this.onFocus)),
					e.off("scroll", this.onScroll)
				}
			},
			disable: function() {
				this.completion.cm.removeKeyMap(this.keyMap);
				var e = this;
				this.keyMap = {
					Enter: function() {
						e.picked = !0
					}
				},
				this.completion.cm.addKeyMap(this.keyMap)
			},
			pick: function() {
				this.completion.pick(this.data, this.selectedHint)
			},
			changeActive: function(t, n) {
				if (t >= this.data.list.length ? t = n ? this.data.list.length - 1 : 0 : 0 > t && (t = n ? 0 : this.data.list.length - 1), this.selectedHint != t) {
					var i = this.hints.childNodes[this.selectedHint];
					i.className = i.className.replace(" " + u, ""),
					(i = this.hints.childNodes[this.selectedHint = t]).className += " " + u,
					i.offsetTop < this.hints.scrollTop ? this.hints.scrollTop = i.offsetTop - 3 : i.offsetTop + i.offsetHeight > this.hints.scrollTop + this.hints.clientHeight && (this.hints.scrollTop = i.offsetTop + i.offsetHeight - this.hints.clientHeight + 3),
					e.signal(this.data, "select", this.data.list[this.selectedHint], i)
				}
			},
			screenAmount: function() {
				return Math.floor(this.hints.clientHeight / this.hints.firstChild.offsetHeight) || 1
			}
		},
		e.registerHelper("hint", "auto", {
			resolve: function(t, n) {
				var i, r = t.getHelpers(n, "hint");
				if (r.length) {
					for (var o, a = !1,
					s = 0; s < r.length; s++) r[s].async && (a = !0);
					return a ? (o = function(e, t, n) {
						function i(r, a) {
							if (r == o.length) return t(null);
							var s = o[r];
							s.async ? s(e,
							function(e) {
								e ? t(e) : i(r + 1)
							},
							n) : (a = s(e, n)) ? t(a) : i(r + 1)
						}
						var o = l(e, r);
						i(0)
					},
					o.async = !0) : o = function(e, t) {
						for (var n = l(e, r), i = 0; i < n.length; i++) {
							var o = n[i](e, t);
							if (o && o.list.length) return o
						}
					},
					o.supportsSelection = !0,
					o
				}
				return (i = t.getHelper(t.getCursor(), "hintWords")) ?
				function(t) {
					return e.hint.fromList(t, {
						words: i
					})
				}: e.hint.anyword ?
				function(t, n) {
					return e.hint.anyword(t, n)
				}: function() {}
			}
		}),
		e.registerHelper("hint", "fromList",
		function(t, n) {
			var i = t.getCursor(),
			r = t.getTokenAt(i),
			o = e.Pos(i.line, r.end);
			if (r.string && /\w/.test(r.string[r.string.length - 1])) var a = r.string,
			s = e.Pos(i.line, r.start);
			else var a = "",
			s = o;
			for (var l = [], c = 0; c < n.words.length; c++) {
				var u = n.words[c];
				u.slice(0, a.length) == a && l.push(u)
			}
			return l.length ? {
				list: l,
				from: s,
				to: o
			}: void 0
		}),
		e.commands.autocomplete = e.showHint;
		var f = {
			hint: e.hint.auto,
			completeSingle: !0,
			alignWithWord: !0,
			closeCharacters: /[\s()\[\]{};:>,]/,
			closeOnUnfocus: !0,
			completeOnSingleClick: !0,
			container: null,
			customKeys: null,
			extraKeys: null
		};
		e.defineOption("hintOptions", null)
	}),
	function(e) {
		"object" == typeof exports && "object" == typeof module ? e(require("../../lib/codemirror"), require("../../mode/sql/sql")) : "function" == typeof define && define.amd ? define(["../../lib/codemirror", "../../mode/sql/sql"], e) : e(CodeMirror)
	} (function(e) {
		"use strict";
		function t(e) {
			return "[object Array]" == Object.prototype.toString.call(e)
		}
		function n(t) {
			var n = t.doc.modeOption;
			return "sql" === n && (n = "text/x-sql"),
			e.resolveMode(n).keywords
		}
		function i(e) {
			return "string" == typeof e ? e: e.text
		}
		function r(e, n) {
			return t(n) && (n = {
				columns: n
			}),
			n.text || (n.text = e),
			n
		}
		function o(e) {
			var n = {};
			if (t(e)) for (var o = e.length - 1; o >= 0; o--) {
				var a = e[o];
				n[i(a).toUpperCase()] = r(i(a), a)
			} else if (e) for (var s in e) n[s.toUpperCase()] = r(s, e[s]);
			return n
		}
		function a(e) {
			return v[e.toUpperCase()]
		}
		function s(e) {
			var t = {};
			for (var n in e) e.hasOwnProperty(n) && (t[n] = e[n]);
			return t
		}
		function l(e, t) {
			var n = e.length,
			r = i(t).substr(0, n);
			return e.toUpperCase() === r.toUpperCase()
		}
		function c(e, n, i, r) {
			if (t(i)) for (var o = 0; o < i.length; o++) l(n, i[o]) && e.push(r(i[o]));
			else for (var a in i) if (i.hasOwnProperty(a)) {
				var s = i[a];
				l(n, s = s && !0 !== s ? s.displayText ? {
					text: s.text,
					displayText: s.displayText
				}: s.text: a) && e.push(r(s))
			}
		}
		function u(e) {
			return "." == e.charAt(0) && (e = e.substr(1)),
			e.replace(/`/g, "")
		}
		function d(e) {
			for (var t = i(e).split("."), n = 0; n < t.length; n++) t[n] = "`" + t[n] + "`";
			var r = t.join(".");
			return "string" == typeof e ? r: (e = s(e), e.text = r, e)
		}
		function h(e, t, n, i) {
			for (var r = !1,
			o = [], l = t.start, h = !0; h;) h = "." == t.string.charAt(0),
			r = r || "`" == t.string.charAt(0),
			l = t.start,
			o.unshift(u(t.string)),
			"." == (t = i.getTokenAt(k(e.line, t.start))).string && (h = !0, t = i.getTokenAt(k(e.line, t.start)));
			var f = o.join(".");
			c(n, f, v,
			function(e) {
				return r ? d(e) : e
			}),
			c(n, f, y,
			function(e) {
				return r ? d(e) : e
			}),
			f = o.pop();
			var p = o.join("."),
			m = !1,
			b = p;
			if (!a(p)) {
				var w = p; (p = g(p, i)) !== w && (m = !0)
			}
			var x = a(p);
			return x && x.columns && (x = x.columns),
			x && c(n, f, x,
			function(e) {
				var t = p;
				return 1 == m && (t = b),
				"string" == typeof e ? e = t + "." + e: (e = s(e), e.text = t + "." + e.text),
				r ? d(e) : e
			}),
			l
		}
		function f(e, t) {
			if (e) for (var n = /[,;]/g,
			i = e.split(" "), r = 0; r < i.length; r++) t(i[r] ? i[r].replace(n, "") : "")
		}
		function p(e) {
			return e.line + e.ch / Math.pow(10, 6)
		}
		function m(e) {
			return k(Math.floor(e), +e.toString().split(".").pop())
		}
		function g(e, t) {
			for (var n = t.doc,
			i = n.getValue(), r = e.toUpperCase(), o = "", s = "", l = [], c = {
				start: k(0, 0),
				end: k(t.lastLine(), t.getLineHandle(t.lastLine()).length)
			},
			u = i.indexOf(w.QUERY_DIV); - 1 != u;) l.push(n.posFromIndex(u)),
			u = i.indexOf(w.QUERY_DIV, u + 1);
			l.unshift(k(0, 0)),
			l.push(k(t.lastLine(), t.getLineHandle(t.lastLine()).text.length));
			for (var d = 0,
			h = p(t.getCursor()), g = 0; g < l.length; g++) {
				var v = p(l[g]);
				if (h > d && v >= h) {
					c = {
						start: m(d),
						end: m(v)
					};
					break
				}
				d = v
			}
			for (var y = n.getRange(c.start, c.end, !1), g = 0; g < y.length && (f(y[g],
			function(e) {
				var t = e.toUpperCase();
				t === r && a(o) && (s = o),
				t !== w.ALIAS_KEYWORD && (o = e)
			}), !s); g++);
			return s
		}
		var v, y, b, w = {
			QUERY_DIV: ";",
			ALIAS_KEYWORD: "AS"
		},
		k = e.Pos;
		e.registerHelper("hint", "sql",
		function(e, t) {
			v = o(t && t.tables);
			var i = t && t.defaultTable,
			r = t && t.disableKeywords;
			y = i && a(i),
			b = b || n(e),
			i && !y && (y = g(i, e)),
			(y = y || []).columns && (y = y.columns);
			var s, l, u, d = e.getCursor(),
			f = [],
			p = e.getTokenAt(d);
			return p.end > d.ch && (p.end = d.ch, p.string = p.string.slice(0, d.ch - p.start)),
			p.string.match(/^[.`\w@]\w*$/) ? (u = p.string, s = p.start, l = p.end) : (s = l = d.ch, u = ""),
			"." == u.charAt(0) || "`" == u.charAt(0) ? s = h(d, p, f, e) : (c(f, u, v,
			function(e) {
				return e
			}), c(f, u, y,
			function(e) {
				return e
			}), r || c(f, u, b,
			function(e) {
				return e.toUpperCase()
			})),
			{
				list: f,
				from: k(d.line, s),
				to: k(d.line, l)
			}
		})
	}),
	function(e) {
		"object" == typeof exports && "object" == typeof module ? e(require("../../lib/codemirror")) : "function" == typeof define && define.amd ? define(["../../lib/codemirror"], e) : e(CodeMirror)
	} (function(e) {
		"use strict";
		function t(e, t) {
			return e.line - t.line || e.ch - t.ch
		}
		function n(e, t, n, i) {
			this.line = t,
			this.ch = n,
			this.cm = e,
			this.text = e.getLine(t),
			this.min = i ? i.from: e.firstLine(),
			this.max = i ? i.to - 1 : e.lastLine()
		}
		function i(e, t) {
			var n = e.cm.getTokenTypeAt(h(e.line, t));
			return n && /\btag\b/.test(n)
		}
		function r(e) {
			return e.line >= e.max ? void 0 : (e.ch = 0, e.text = e.cm.getLine(++e.line), !0)
		}
		function o(e) {
			return e.line <= e.min ? void 0 : (e.text = e.cm.getLine(--e.line), e.ch = e.text.length, !0)
		}
		function a(e) {
			for (;;) {
				var t = e.text.indexOf(">", e.ch);
				if ( - 1 == t) {
					if (r(e)) continue;
					return
				}
				if (i(e, t + 1)) {
					var n = e.text.lastIndexOf("/", t),
					o = n > -1 && !/\S/.test(e.text.slice(n + 1, t));
					return e.ch = t + 1,
					o ? "selfClose": "regular"
				}
				e.ch = t + 1
			}
		}
		function s(e) {
			for (;;) {
				var t = e.ch ? e.text.lastIndexOf("<", e.ch - 1) : -1;
				if ( - 1 == t) {
					if (o(e)) continue;
					return
				}
				if (i(e, t + 1)) {
					p.lastIndex = t,
					e.ch = t;
					var n = p.exec(e.text);
					if (n && n.index == t) return n
				} else e.ch = t
			}
		}
		function l(e) {
			for (;;) {
				p.lastIndex = e.ch;
				var t = p.exec(e.text);
				if (!t) {
					if (r(e)) continue;
					return
				}
				if (i(e, t.index + 1)) return e.ch = t.index + t[0].length,
				t;
				e.ch = t.index + 1
			}
		}
		function c(e) {
			for (;;) {
				var t = e.ch ? e.text.lastIndexOf(">", e.ch - 1) : -1;
				if ( - 1 == t) {
					if (o(e)) continue;
					return
				}
				if (i(e, t + 1)) {
					var n = e.text.lastIndexOf("/", t),
					r = n > -1 && !/\S/.test(e.text.slice(n + 1, t));
					return e.ch = t + 1,
					r ? "selfClose": "regular"
				}
				e.ch = t
			}
		}
		function u(e, t) {
			for (var n = [];;) {
				var i, r = l(e),
				o = e.line,
				s = e.ch - (r ? r[0].length: 0);
				if (!r || !(i = a(e))) return;
				if ("selfClose" != i) if (r[1]) {
					for (var c = n.length - 1; c >= 0; --c) if (n[c] == r[2]) {
						n.length = c;
						break
					}
					if (0 > c && (!t || t == r[2])) return {
						tag: r[2],
						from: h(o, s),
						to: h(e.line, e.ch)
					}
				} else n.push(r[2])
			}
		}
		function d(e, t) {
			for (var n = [];;) {
				var i = c(e);
				if (!i) return;
				if ("selfClose" != i) {
					var r = e.line,
					o = e.ch,
					a = s(e);
					if (!a) return;
					if (a[1]) n.push(a[2]);
					else {
						for (var l = n.length - 1; l >= 0; --l) if (n[l] == a[2]) {
							n.length = l;
							break
						}
						if (0 > l && (!t || t == a[2])) return {
							tag: a[2],
							from: h(e.line, e.ch),
							to: h(r, o)
						}
					}
				} else s(e)
			}
		}
		var h = e.Pos,
		f = "A-Z_a-z\\u00C0-\\u00D6\\u00D8-\\u00F6\\u00F8-\\u02FF\\u0370-\\u037D\\u037F-\\u1FFF\\u200C-\\u200D\\u2070-\\u218F\\u2C00-\\u2FEF\\u3001-\\uD7FF\\uF900-\\uFDCF\\uFDF0-\\uFFFD",
		p = new RegExp("<(/?)([" + f + "][A-Z_a-z\\u00C0-\\u00D6\\u00D8-\\u00F6\\u00F8-\\u02FF\\u0370-\\u037D\\u037F-\\u1FFF\\u200C-\\u200D\\u2070-\\u218F\\u2C00-\\u2FEF\\u3001-\\uD7FF\\uF900-\\uFDCF\\uFDF0-\\uFFFD-:.0-9\\u00B7\\u0300-\\u036F\\u203F-\\u2040]*)", "g");
		e.registerHelper("fold", "xml",
		function(e, t) {
			for (var i = new n(e, t.line, 0);;) {
				var r, o = l(i);
				if (!o || i.line != t.line || !(r = a(i))) return;
				if (!o[1] && "selfClose" != r) {
					var t = h(i.line, i.ch),
					s = u(i, o[2]);
					return s && {
						from: t,
						to: s.from
					}
				}
			}
		}),
		e.findMatchingTag = function(e, i, r) {
			var o = new n(e, i.line, i.ch, r);
			if ( - 1 != o.text.indexOf(">") || -1 != o.text.indexOf("<")) {
				var l = a(o),
				c = l && h(o.line, o.ch),
				f = l && s(o);
				if (l && f && !(t(o, i) > 0)) {
					var p = {
						from: h(o.line, o.ch),
						to: c,
						tag: f[2]
					};
					return "selfClose" == l ? {
						open: p,
						close: null,
						at: "open"
					}: f[1] ? {
						open: d(o, f[2]),
						close: p,
						at: "close"
					}: (o = new n(e, c.line, c.ch, r), {
						open: p,
						close: u(o, f[2]),
						at: "open"
					})
				}
			}
		},
		e.findEnclosingTag = function(e, t, i) {
			for (var r = new n(e, t.line, t.ch, i);;) {
				var o = d(r);
				if (!o) break;
				var a = u(new n(e, t.line, t.ch, i), o.tag);
				if (a) return {
					open: o,
					close: a
				}
			}
		},
		e.scanForClosingTag = function(e, t, i, r) {
			return u(new n(e, t.line, t.ch, r ? {
				from: 0,
				to: r
			}: null), i)
		}
	}),
	function(e) {
		"object" == typeof exports && "object" == typeof module ? e(require("../../lib/codemirror")) : "function" == typeof define && define.amd ? define(["../../lib/codemirror"], e) : e(CodeMirror)
	} (function(e) {
		"use strict";
		var t = e.Pos;
		e.registerHelper("hint", "xml",
		function(n, i) {
			var r = i && i.schemaInfo,
			o = i && i.quoteChar || '"';
			if (r) {
				var a = n.getCursor(),
				s = n.getTokenAt(a);
				s.end > a.ch && (s.end = a.ch, s.string = s.string.slice(0, a.ch - s.start));
				var l = e.innerMode(n.getMode(), s.state);
				if ("xml" == l.mode.name) {
					var c, u, d = [],
					h = !1,
					f = /\btag\b/.test(s.type) && !/>$/.test(s.string),
					p = f && /^\w/.test(s.string);
					if (p) {
						var m = n.getLine(a.line).slice(Math.max(0, s.start - 2), s.start),
						g = /<\/$/.test(m) ? "close": /<$/.test(m) ? "open": null;
						g && (u = s.start - ("close" == g ? 2 : 1))
					} else f && "<" == s.string ? g = "open": f && "</" == s.string && (g = "close");
					if (!f && !l.state.tagName || g) {
						p && (c = s.string),
						h = g;
						var v = l.state.context,
						y = v && r[v.tagName],
						b = v ? y && y.children: r["!top"];
						if (b && "close" != g) for (L = 0; L < b.length; ++L) c && 0 != b[L].lastIndexOf(c, 0) || d.push("<" + b[L]);
						else if ("close" != g) for (var w in r) ! r.hasOwnProperty(w) || "!top" == w || "!attrs" == w || c && 0 != w.lastIndexOf(c, 0) || d.push("<" + w);
						v && (!c || "close" == g && 0 == v.tagName.lastIndexOf(c, 0)) && d.push("</" + v.tagName + ">")
					} else {
						var k = (y = r[l.state.tagName]) && y.attrs,
						x = r["!attrs"];
						if (!k && !x) return;
						if (k) {
							if (x) {
								var _ = {};
								for (var C in x) x.hasOwnProperty(C) && (_[C] = x[C]);
								for (var C in k) k.hasOwnProperty(C) && (_[C] = k[C]);
								k = _
							}
						} else k = x;
						if ("string" == s.type || "=" == s.string) {
							var S, M = (m = n.getRange(t(a.line, Math.max(0, a.ch - 60)), t(a.line, "string" == s.type ? s.start: s.end))).match(/([^\s\u00a0=<>\"\']+)=$/);
							if (!M || !k.hasOwnProperty(M[1]) || !(S = k[M[1]])) return;
							if ("function" == typeof S && (S = S.call(this, n)), "string" == s.type) {
								c = s.string;
								var T = 0;
								/['"]/.test(s.string.charAt(0)) && (o = s.string.charAt(0), c = s.string.slice(1), T++);
								var D = s.string.length;
								/['"]/.test(s.string.charAt(D - 1)) && (o = s.string.charAt(D - 1), c = s.string.substr(T, D - 2)),
								h = !0
							}
							for (var L = 0; L < S.length; ++L) c && 0 != S[L].lastIndexOf(c, 0) || d.push(o + S[L] + o)
						} else {
							"attribute" == s.type && (c = s.string, h = !0);
							for (var O in k) ! k.hasOwnProperty(O) || c && 0 != O.lastIndexOf(c, 0) || d.push(O)
						}
					}
					return {
						list: d,
						from: h ? t(a.line, null == u ? s.start: u) : a,
						to: h ? t(a.line, s.end) : a
					}
				}
			}
		})
	});
	var hljs = new
	function() {
		function e(e) {
			return e.replace(/&/gm, "&amp;").replace(/</gm, "&lt;").replace(/>/gm, "&gt;")
		}
		function t(e) {
			for (var t = e.firstChild; t; t = t.nextSibling) {
				if ("CODE" == t.nodeName) return t;
				if (3 != t.nodeType || !t.nodeValue.match(/\s+/)) break
			}
		}
		function n(e, t) {
			return Array.prototype.map.call(e.childNodes,
			function(e) {
				return 3 == e.nodeType ? t ? e.nodeValue.replace(/\n/g, "") : e.nodeValue: "BR" == e.nodeName ? "\n": n(e, t)
			}).join("")
		}
		function i(e) {
			var t = (e.className + " " + e.parentNode.className).split(/\s+/);
			t = t.map(function(e) {
				return e.replace(/^language-/, "")
			});
			for (var n = 0; n < t.length; n++) if (h[t[n]] || "no-highlight" == t[n]) return t[n]
		}
		function r(e) {
			var t = [];
			return function e(n, i) {
				for (var r = n.firstChild; r; r = r.nextSibling) 3 == r.nodeType ? i += r.nodeValue.length: "BR" == r.nodeName ? i += 1 : 1 == r.nodeType && (t.push({
					event: "start",
					offset: i,
					node: r
				}), i = e(r, i), t.push({
					event: "stop",
					offset: i,
					node: r
				}));
				return i
			} (e, 0),
			t
		}
		function o(t, n, i) {
			function r(t) {
				return "<" + t.nodeName + Array.prototype.map.call(t.attributes,
				function(t) {
					return " " + t.nodeName + '="' + e(t.value) + '"'
				}).join("") + ">"
			}
			for (var o = 0,
			a = "",
			s = []; t.length || n.length;) {
				var l = (t.length && n.length ? t[0].offset != n[0].offset ? t[0].offset < n[0].offset ? t: n: "start" == n[0].event ? t: n: t.length ? t: n).splice(0, 1)[0];
				if (a += e(i.substr(o, l.offset - o)), o = l.offset, "start" == l.event) a += r(l.node),
				s.push(l.node);
				else if ("stop" == l.event) {
					var c, u = s.length;
					do {
						a += "</" + (c = s[--u]).nodeName.toLowerCase() + ">"
					} while ( c != l . node );
					for (s.splice(u, 1); u < s.length;) a += r(s[u]),
					u++
				}
			}
			return a + e(i.substr(o))
		}
		function a(e) {
			function t(t, n) {
				return RegExp(t, "m" + (e.cI ? "i": "") + (n ? "g": ""))
			}
			function n(e, i) {
				function r(e, t) {
					t.split(" ").forEach(function(t) {
						var n = t.split("|");
						a[n[0]] = [e, n[1] ? Number(n[1]) : 1],
						o.push(n[0])
					})
				}
				if (!e.compiled) {
					e.compiled = !0;
					var o = [];
					if (e.k) {
						var a = {};
						if (e.lR = t(e.l || hljs.IR, !0), "string" == typeof e.k) r("keyword", e.k);
						else for (var s in e.k) e.k.hasOwnProperty(s) && r(s, e.k[s]);
						e.k = a
					}
					i && (e.bWK && (e.b = "\\b(" + o.join("|") + ")\\s"), e.bR = t(e.b ? e.b: "\\B|\\b"), e.e || e.eW || (e.e = "\\B|\\b"), e.e && (e.eR = t(e.e)), e.tE = e.e || "", e.eW && i.tE && (e.tE += (e.e ? "|": "") + i.tE)),
					e.i && (e.iR = t(e.i)),
					void 0 === e.r && (e.r = 1),
					e.c || (e.c = []);
					for (c = 0; c < e.c.length; c++)"self" == e.c[c] && (e.c[c] = e),
					n(e.c[c], e);
					e.starts && n(e.starts, i);
					for (var l = [], c = 0; c < e.c.length; c++) l.push(e.c[c].b);
					e.tE && l.push(e.tE),
					e.i && l.push(e.i),
					e.t = l.length ? t(l.join("|"), !0) : {
						exec: function(e) {
							return null
						}
					}
				}
			}
			n(e)
		}
		function s(t, n) {
			function i(e, t) {
				for (var n = 0; n < t.c.length; n++) {
					var i = t.c[n].bR.exec(e);
					if (i && 0 == i.index) return t.c[n]
				}
			}
			function r(e, t) {
				return e.e && e.eR.test(t) ? e: e.eW ? r(e.parent, t) : void 0
			}
			function o(e, t) {
				return t.i && t.iR.test(e)
			}
			function c(e, t) {
				var n = g.cI ? t[0].toLowerCase() : t[0];
				return e.k.hasOwnProperty(n) && e.k[n]
			}
			function u() {
				var t = e(y);
				if (!v.k) return t;
				var n = "",
				i = 0;
				v.lR.lastIndex = 0;
				for (var r = v.lR.exec(t); r;) {
					n += t.substr(i, r.index - i);
					var o = c(v, r);
					o ? (w += o[1], n += '<span class="' + o[0] + '">' + r[0] + "</span>") : n += r[0],
					i = v.lR.lastIndex,
					r = v.lR.exec(t)
				}
				return n + t.substr(i)
			}
			function d() {
				if (v.sL && !h[v.sL]) return e(y);
				var t = v.sL ? s(v.sL, y) : l(y);
				return v.r > 0 && (w += t.keyword_count, b += t.r),
				'<span class="' + t.language + '">' + t.value + "</span>"
			}
			function f() {
				return void 0 !== v.sL ? d() : u()
			}
			function p(t, n) {
				var i = t.cN ? '<span class="' + t.cN + '">': "";
				t.rB ? (k += i, y = "") : t.eB ? (k += e(n) + i, y = "") : (k += i, y = n),
				v = Object.create(t, {
					parent: {
						value: v
					}
				}),
				b += t.r
			}
			function m(t, n) {
				if (y += t, void 0 === n) return k += f(),
				0;
				var a = i(n, v);
				if (a) return k += f(),
				p(a, n),
				a.rB ? 0 : n.length;
				var s = r(v, n);
				if (s) {
					s.rE || s.eE || (y += n),
					k += f();
					do {
						v.cN && (k += "</span>"), v = v.parent
					} while ( v != s . parent );
					return s.eE && (k += e(n)),
					y = "",
					s.starts && p(s.starts, ""),
					s.rE ? 0 : n.length
				}
				if (o(n, v)) throw "Illegal";
				return y += n,
				n.length || 1
			}
			var g = h[t];
			a(g);
			var v = g,
			y = "",
			b = 0,
			w = 0,
			k = "";
			try {
				for (var x, _, C = 0;;) {
					if (v.t.lastIndex = C, !(x = v.t.exec(n))) break;
					_ = m(n.substr(C, x.index - C), x[0]),
					C = x.index + _
				}
				return m(n.substr(C)),
				{
					r: b,
					keyword_count: w,
					value: k,
					language: t
				}
			} catch(t) {
				if ("Illegal" == t) return {
					r: 0,
					keyword_count: 0,
					value: e(n)
				};
				throw t
			}
		}
		function l(t) {
			var n = {
				keyword_count: 0,
				r: 0,
				value: e(t)
			},
			i = n;
			for (var r in h) if (h.hasOwnProperty(r)) {
				var o = s(r, t);
				o.language = r,
				o.keyword_count + o.r > i.keyword_count + i.r && (i = o),
				o.keyword_count + o.r > n.keyword_count + n.r && (i = n, n = o)
			}
			return i.language && (n.second_best = i),
			n
		}
		function c(e, t, n) {
			return t && (e = e.replace(/^((<[^>]+>|\t)+)/gm,
			function(e, n, i, r) {
				return n.replace(/\t/g, t)
			})),
			n && (e = e.replace(/\n/g, "<br>")),
			e
		}
		function u(e, t, a) {
			var u = n(e, a),
			d = i(e);
			if ("no-highlight" != d) {
				var h = d ? s(d, u) : l(u);
				d = h.language;
				var f = r(e);
				if (f.length) {
					var p = document.createElement("pre");
					p.innerHTML = h.value,
					h.value = o(f, r(p), u)
				}
				h.value = c(h.value, t, a);
				var m = e.className;
				m.match("(\\s|^)(language-)?" + d + "(\\s|$)") || (m = m ? m + " " + d: d),
				e.innerHTML = h.value,
				e.className = m,
				e.result = {
					language: d,
					kw: h.keyword_count,
					re: h.r
				},
				h.second_best && (e.second_best = {
					language: h.second_best.language,
					kw: h.second_best.keyword_count,
					re: h.second_best.r
				})
			}
		}
		function d() {
			d.called || (d.called = !0, Array.prototype.map.call(document.getElementsByTagName("pre"), t).filter(Boolean).forEach(function(e) {
				u(e, hljs.tabReplace)
			}))
		}
		var h = {};
		this.LANGUAGES = h,
		this.highlight = s,
		this.highlightAuto = l,
		this.fixMarkup = c,
		this.highlightBlock = u,
		this.initHighlighting = d,
		this.initHighlightingOnLoad = function() {
			window.addEventListener("DOMContentLoaded", d, !1),
			window.addEventListener("load", d, !1)
		},
		this.IR = "[a-zA-Z][a-zA-Z0-9_]*",
		this.UIR = "[a-zA-Z_][a-zA-Z0-9_]*",
		this.NR = "\\b\\d+(\\.\\d+)?",
		this.CNR = "(\\b0[xX][a-fA-F0-9]+|(\\b\\d+(\\.\\d*)?|\\.\\d+)([eE][-+]?\\d+)?)",
		this.BNR = "\\b(0b[01]+)",
		this.RSR = "!|!=|!==|%|%=|&|&&|&=|\\*|\\*=|\\+|\\+=|,|\\.|-|-=|/|/=|:|;|<|<<|<<=|<=|=|==|===|>|>=|>>|>>=|>>>|>>>=|\\?|\\[|\\{|\\(|\\^|\\^=|\\||\\|=|\\|\\||~",
		this.BE = {
			b: "\\\\[\\s\\S]",
			r: 0
		},
		this.ASM = {
			cN: "string",
			b: "'",
			e: "'",
			i: "\\n",
			c: [this.BE],
			r: 0
		},
		this.QSM = {
			cN: "string",
			b: '"',
			e: '"',
			i: "\\n",
			c: [this.BE],
			r: 0
		},
		this.CLCM = {
			cN: "comment",
			b: "//",
			e: "$"
		},
		this.CBLCLM = {
			cN: "comment",
			b: "/\\*",
			e: "\\*/"
		},
		this.HCM = {
			cN: "comment",
			b: "#",
			e: "$"
		},
		this.NM = {
			cN: "number",
			b: this.NR,
			r: 0
		},
		this.CNM = {
			cN: "number",
			b: this.CNR,
			r: 0
		},
		this.BNM = {
			cN: "number",
			b: this.BNR,
			r: 0
		},
		this.inherit = function(e, t) {
			var n = {};
			for (var i in e) n[i] = e[i];
			if (t) for (var i in t) n[i] = t[i];
			return n
		}
	}; hljs.LANGUAGES.ruby = function(e) {
		var t = "[a-zA-Z_][a-zA-Z0-9_]*(\\!|\\?)?",
		n = "[a-zA-Z_]\\w*[!?=]?|[-+~]\\@|<<|>>|=~|===?|<=>|[<>]=?|\\*\\*|[-/+%^&*~`|]|\\[\\]=?",
		i = {
			keyword: "and false then defined module in return redo if BEGIN retry end for true self when next until do begin unless END rescue nil else break undef not super class case require yield alias while ensure elsif or include"
		},
		r = {
			cN: "yardoctag",
			b: "@[A-Za-z]+"
		},
		o = [{
			cN: "comment",
			b: "#",
			e: "$",
			c: [r]
		},
		{
			cN: "comment",
			b: "^\\=begin",
			e: "^\\=end",
			c: [r],
			r: 10
		},
		{
			cN: "comment",
			b: "^__END__",
			e: "\\n$"
		}],
		a = {
			cN: "subst",
			b: "#\\{",
			e: "}",
			l: t,
			k: i
		},
		s = [e.BE, a],
		l = [{
			cN: "string",
			b: "'",
			e: "'",
			c: s,
			r: 0
		},
		{
			cN: "string",
			b: '"',
			e: '"',
			c: s,
			r: 0
		},
		{
			cN: "string",
			b: "%[qw]?\\(",
			e: "\\)",
			c: s
		},
		{
			cN: "string",
			b: "%[qw]?\\[",
			e: "\\]",
			c: s
		},
		{
			cN: "string",
			b: "%[qw]?{",
			e: "}",
			c: s
		},
		{
			cN: "string",
			b: "%[qw]?<",
			e: ">",
			c: s,
			r: 10
		},
		{
			cN: "string",
			b: "%[qw]?/",
			e: "/",
			c: s,
			r: 10
		},
		{
			cN: "string",
			b: "%[qw]?%",
			e: "%",
			c: s,
			r: 10
		},
		{
			cN: "string",
			b: "%[qw]?-",
			e: "-",
			c: s,
			r: 10
		},
		{
			cN: "string",
			b: "%[qw]?\\|",
			e: "\\|",
			c: s,
			r: 10
		}],
		c = {
			cN: "function",
			bWK: !0,
			e: " |$|;",
			k: "def",
			c: [{
				cN: "title",
				b: n,
				l: t,
				k: i
			},
			{
				cN: "params",
				b: "\\(",
				e: "\\)",
				l: t,
				k: i
			}].concat(o)
		},
		u = o.concat(l.concat([{
			cN: "class",
			bWK: !0,
			e: "$|;",
			k: "class module",
			c: [{
				cN: "title",
				b: "[A-Za-z_]\\w*(::\\w+)*(\\?|\\!)?",
				r: 0
			},
			{
				cN: "inheritance",
				b: "<\\s*",
				c: [{
					cN: "parent",
					b: "(" + e.IR + "::)?" + e.IR
				}]
			}].concat(o)
		},
		c, {
			cN: "constant",
			b: "(::)?(\\b[A-Z]\\w*(::)?)+",
			r: 0
		},
		{
			cN: "symbol",
			b: ":",
			c: l.concat([{
				b: n
			}]),
			r: 0
		},
		{
			cN: "symbol",
			b: t + ":",
			r: 0
		},
		{
			cN: "number",
			b: "(\\b0[0-7_]+)|(\\b0x[0-9a-fA-F_]+)|(\\b[1-9][0-9_]*(\\.[0-9_]+)?)|[0_]\\b",
			r: 0
		},
		{
			cN: "number",
			b: "\\?\\w"
		},
		{
			cN: "variable",
			b: "(\\$\\W)|((\\$|\\@\\@?)(\\w+))"
		},
		{
			b: "(" + e.RSR + ")\\s*",
			c: o.concat([{
				cN: "regexp",
				b: "/",
				e: "/[a-z]*",
				i: "\\n",
				c: [e.BE, a]
			}]),
			r: 0
		}]));
		return a.c = u,
		c.c[1].c = u,
		{
			l: t,
			k: i,
			c: u
		}
	} (hljs), hljs.LANGUAGES.javascript = function(e) {
		return {
			k: {
				keyword: "in if for while finally var new function do return void else break catch instanceof with throw case default try this switch continue typeof delete let yield const",
				literal: "true false null undefined NaN Infinity"
			},
			c: [e.ASM, e.QSM, e.CLCM, e.CBLCLM, e.CNM, {
				b: "(" + e.RSR + "|\\b(case|return|throw)\\b)\\s*",
				k: "return throw case",
				c: [e.CLCM, e.CBLCLM, {
					cN: "regexp",
					b: "/",
					e: "/[gim]*",
					i: "\\n",
					c: [{
						b: "\\\\/"
					}]
				},
				{
					b: "<",
					e: ">;",
					sL: "xml"
				}],
				r: 0
			},
			{
				cN: "function",
				bWK: !0,
				e: "{",
				k: "function",
				c: [{
					cN: "title",
					b: "[A-Za-z$_][0-9A-Za-z$_]*"
				},
				{
					cN: "params",
					b: "\\(",
					e: "\\)",
					c: [e.CLCM, e.CBLCLM],
					i: "[\"'\\(]"
				}],
				i: "\\[|%"
			}]
		}
	} (hljs), hljs.LANGUAGES.css = function(e) {
		var t = {
			cN: "function",
			b: e.IR + "\\(",
			e: "\\)",
			c: [e.NM, e.ASM, e.QSM]
		};
		return {
			cI: !0,
			i: "[=/|']",
			c: [e.CBLCLM, {
				cN: "id",
				b: "\\#[A-Za-z0-9_-]+"
			},
			{
				cN: "class",
				b: "\\.[A-Za-z0-9_-]+",
				r: 0
			},
			{
				cN: "attr_selector",
				b: "\\[",
				e: "\\]",
				i: "$"
			},
			{
				cN: "pseudo",
				b: ":(:)?[a-zA-Z0-9\\_\\-\\+\\(\\)\\\"\\']+"
			},
			{
				cN: "at_rule",
				b: "@(font-face|page)",
				l: "[a-z-]+",
				k: "font-face page"
			},
			{
				cN: "at_rule",
				b: "@",
				e: "[{;]",
				eE: !0,
				k: "import page media charset",
				c: [t, e.ASM, e.QSM, e.NM]
			},
			{
				cN: "tag",
				b: e.IR,
				r: 0
			},
			{
				cN: "rules",
				b: "{",
				e: "}",
				i: "[^\\s]",
				r: 0,
				c: [e.CBLCLM, {
					cN: "rule",
					b: "[^\\s]",
					rB: !0,
					e: ";",
					eW: !0,
					c: [{
						cN: "attribute",
						b: "[A-Z\\_\\.\\-]+",
						e: ":",
						eE: !0,
						i: "[^\\s]",
						starts: {
							cN: "value",
							eW: !0,
							eE: !0,
							c: [t, e.NM, e.QSM, e.ASM, e.CBLCLM, {
								cN: "hexcolor",
								b: "\\#[0-9A-F]+"
							},
							{
								cN: "important",
								b: "!important"
							}]
						}
					}]
				}]
			}]
		}
	} (hljs), hljs.LANGUAGES.xml = function(e) {
		var t = {
			eW: !0,
			c: [{
				cN: "attribute",
				b: "[A-Za-z0-9\\._:-]+",
				r: 0
			},
			{
				b: '="',
				rB: !0,
				e: '"',
				c: [{
					cN: "value",
					b: '"',
					eW: !0
				}]
			},
			{
				b: "='",
				rB: !0,
				e: "'",
				c: [{
					cN: "value",
					b: "'",
					eW: !0
				}]
			},
			{
				b: "=",
				c: [{
					cN: "value",
					b: "[^\\s/>]+"
				}]
			}]
		};
		return {
			cI: !0,
			c: [{
				cN: "pi",
				b: "<\\?",
				e: "\\?>",
				r: 10
			},
			{
				cN: "doctype",
				b: "<!DOCTYPE",
				e: ">",
				r: 10,
				c: [{
					b: "\\[",
					e: "\\]"
				}]
			},
			{
				cN: "comment",
				b: "\x3c!--",
				e: "--\x3e",
				r: 10
			},
			{
				cN: "cdata",
				b: "<\\!\\[CDATA\\[",
				e: "\\]\\]>",
				r: 10
			},
			{
				cN: "tag",
				b: "<style(?=\\s|>|$)",
				e: ">",
				k: {
					title: "style"
				},
				c: [t],
				starts: {
					e: "</style>",
					rE: !0,
					sL: "css"
				}
			},
			{
				cN: "tag",
				b: "<script(?=\\s|>|$)",
				e: ">",
				k: {
					title: "script"
				},
				c: [t],
				starts: {
					e: "<\/script>",
					rE: !0,
					sL: "javascript"
				}
			},
			{
				b: "<%",
				e: "%>",
				sL: "vbscript"
			},
			{
				cN: "tag",
				b: "</?",
				e: "/?>",
				c: [{
					cN: "title",
					b: "[^ />]+"
				},
				t]
			}]
		}
	} (), hljs.LANGUAGES.java = function(e) {
		return {
			k: "false synchronized int abstract float private char boolean static null if const for true while long throw strictfp finally protected import native final return void enum else break transient new catch instanceof byte super volatile case assert short package default double public try this switch continue throws",
			c: [{
				cN: "javadoc",
				b: "/\\*\\*",
				e: "\\*/",
				c: [{
					cN: "javadoctag",
					b: "@[A-Za-z]+"
				}],
				r: 10
			},
			e.CLCM, e.CBLCLM, e.ASM, e.QSM, {
				cN: "class",
				bWK: !0,
				e: "{",
				k: "class interface",
				i: ":",
				c: [{
					bWK: !0,
					k: "extends implements",
					r: 10
				},
				{
					cN: "title",
					b: e.UIR
				}]
			},
			e.CNM, {
				cN: "annotation",
				b: "@[A-Za-z]+"
			}]
		}
	} (hljs), hljs.LANGUAGES.php = function(e) {
		var t = {
			cN: "variable",
			b: "\\$+[a-zA-Z_-ÿ][a-zA-Z0-9_-ÿ]*"
		},
		n = [e.inherit(e.ASM, {
			i: null
		}), e.inherit(e.QSM, {
			i: null
		}), {
			cN: "string",
			b: 'b"',
			e: '"',
			c: [e.BE]
		},
		{
			cN: "string",
			b: "b'",
			e: "'",
			c: [e.BE]
		}],
		i = [e.BNM, e.CNM],
		r = {
			cN: "title",
			b: e.UIR
		};
		return {
			cI: !0,
			k: "and include_once list abstract global private echo interface as static endswitch array null if endwhile or const for endforeach self var while isset public protected exit foreach throw elseif include __FILE__ empty require_once do xor return implements parent clone use __CLASS__ __LINE__ else break print eval new catch __METHOD__ case exception php_user_filter default die require __FUNCTION__ enddeclare final try this switch continue endfor endif declare unset true false namespace trait goto instanceof insteadof __DIR__ __NAMESPACE__ __halt_compiler",
			c: [e.CLCM, e.HCM, {
				cN: "comment",
				b: "/\\*",
				e: "\\*/",
				c: [{
					cN: "phpdoc",
					b: "\\s@[A-Za-z]+"
				}]
			},
			{
				cN: "comment",
				eB: !0,
				b: "__halt_compiler.+?;",
				eW: !0
			},
			{
				cN: "string",
				b: "<<<['\"]?\\w+['\"]?$",
				e: "^\\w+;",
				c: [e.BE]
			},
			{
				cN: "preprocessor",
				b: "<\\?php",
				r: 10
			},
			{
				cN: "preprocessor",
				b: "\\?>"
			},
			t, {
				cN: "function",
				bWK: !0,
				e: "{",
				k: "function",
				i: "\\$|\\[|%",
				c: [r, {
					cN: "params",
					b: "\\(",
					e: "\\)",
					c: ["self", t, e.CBLCLM].concat(n).concat(i)
				}]
			},
			{
				cN: "class",
				bWK: !0,
				e: "{",
				k: "class",
				i: "[:\\(\\$]",
				c: [{
					bWK: !0,
					eW: !0,
					k: "extends",
					c: [r]
				},
				r]
			},
			{
				b: "=>"
			}].concat(n).concat(i)
		}
	} (hljs), hljs.LANGUAGES.python = function(e) {
		var t = {
			cN: "prompt",
			b: "^(>>>|\\.\\.\\.) "
		},
		n = [{
			cN: "string",
			b: "(u|b)?r?'''",
			e: "'''",
			c: [t],
			r: 10
		},
		{
			cN: "string",
			b: '(u|b)?r?"""',
			e: '"""',
			c: [t],
			r: 10
		},
		{
			cN: "string",
			b: "(u|r|ur)'",
			e: "'",
			c: [e.BE],
			r: 10
		},
		{
			cN: "string",
			b: '(u|r|ur)"',
			e: '"',
			c: [e.BE],
			r: 10
		},
		{
			cN: "string",
			b: "(b|br)'",
			e: "'",
			c: [e.BE]
		},
		{
			cN: "string",
			b: '(b|br)"',
			e: '"',
			c: [e.BE]
		}].concat([e.ASM, e.QSM]),
		i = {
			bWK: !0,
			e: ":",
			i: "[${=;\\n]",
			c: [{
				cN: "title",
				b: e.UIR
			},
			{
				cN: "params",
				b: "\\(",
				e: "\\)",
				c: ["self", e.CNM, t].concat(n)
			}],
			r: 10
		};
		return {
			k: {
				keyword: "and elif is global as in if from raise for except finally print import pass return exec else break not with class assert yield try while continue del or def lambda nonlocal|10",
				built_in: "None True False Ellipsis NotImplemented"
			},
			i: "(</|->|\\?)",
			c: n.concat([t, e.HCM, e.inherit(i, {
				cN: "function",
				k: "def"
			}), e.inherit(i, {
				cN: "class",
				k: "class"
			}), e.CNM, {
				cN: "decorator",
				b: "@",
				e: "$"
			},
			{
				b: "\\b(print|exec)\\("
			}])
		}
	} (hljs), hljs.LANGUAGES.sql = function(e) {
		return {
			cI: !0,
			c: [{
				cN: "operator",
				b: "(begin|start|commit|rollback|savepoint|lock|alter|create|drop|rename|call|delete|do|handler|insert|load|replace|select|truncate|update|set|show|pragma|grant)\\b(?!:)",
				e: ";",
				eW: !0,
				k: {
					keyword: "all partial global month current_timestamp using go revoke smallint indicator end-exec disconnect zone with character assertion to add current_user usage input local alter match collate real then rollback get read timestamp session_user not integer bit unique day minute desc insert execute like ilike|2 level decimal drop continue isolation found where constraints domain right national some module transaction relative second connect escape close system_user for deferred section cast current sqlstate allocate intersect deallocate numeric public preserve full goto initially asc no key output collation group by union session both last language constraint column of space foreign deferrable prior connection unknown action commit view or first into float year primary cascaded except restrict set references names table outer open select size are rows from prepare distinct leading create only next inner authorization schema corresponding option declare precision immediate else timezone_minute external varying translation true case exception join hour default double scroll value cursor descriptor values dec fetch procedure delete and false int is describe char as at in varchar null trailing any absolute current_time end grant privileges when cross check write current_date pad begin temporary exec time update catalog user sql date on identity timezone_hour natural whenever interval work order cascade diagnostics nchar having left call do handler load replace truncate start lock show pragma exists number",
					aggregate: "count sum min max avg"
				},
				c: [{
					cN: "string",
					b: "'",
					e: "'",
					c: [e.BE, {
						b: "''"
					}],
					r: 0
				},
				{
					cN: "string",
					b: '"',
					e: '"',
					c: [e.BE, {
						b: '""'
					}],
					r: 0
				},
				{
					cN: "string",
					b: "`",
					e: "`",
					c: [e.BE]
				},
				e.CNM]
			},
			e.CBLCLM, {
				cN: "comment",
				b: "--",
				e: "$"
			}]
		}
	} (hljs), hljs.LANGUAGES.ini = function(e) {
		return {
			cI: !0,
			i: "[^\\s]",
			c: [{
				cN: "comment",
				b: ";",
				e: "$"
			},
			{
				cN: "title",
				b: "^\\[",
				e: "\\]"
			},
			{
				cN: "setting",
				b: "^[a-z0-9\\[\\]_-]+[ \\t]*=[ \\t]*",
				e: "$",
				c: [{
					cN: "value",
					eW: !0,
					k: "on off true false yes no",
					c: [e.QSM, e.NM]
				}]
			}]
		}
	} (hljs), hljs.LANGUAGES.json = function(e) {
		var t = {
			literal: "true false null"
		},
		n = [e.QSM, e.CNM],
		i = {
			cN: "value",
			e: ",",
			eW: !0,
			eE: !0,
			c: n,
			k: t
		},
		r = {
			b: "{",
			e: "}",
			c: [{
				cN: "attribute",
				b: '\\s*"',
				e: '"\\s*:\\s*',
				eB: !0,
				eE: !0,
				c: [e.BE],
				i: "\\n",
				starts: i
			}],
			i: "\\S"
		},
		o = {
			b: "\\[",
			e: "\\]",
			c: [e.inherit(i, {
				cN: null
			})],
			i: "\\S"
		};
		return n.splice(n.length, 0, r, o),
		{
			c: n,
			k: t,
			i: "\\S"
		}
	} (hljs);
	var initRunCode = function() {
		var e = 0,
		t = function(e) {
			for (var t; e.length > 0 && ("\n" === (t = e[0]) || "\r" === t);) e = e.substring(1);
			for (; e.length > 0 && ("\n" === (t = e[e.length - 1]) || "\r" === t);) e = e.substring(0, e.length - 1);
			return e + "\n"
		};
		return function(n, i) {
			var r = "online-run-code-" + ++e,
			o = n.children("code"),
			a = null,
			s = o.text().split("----", 3);
			o.remove(),
			n.attr("id", "pre-" + r),
			n.css("font-size", "14px"),
			n.css("margin-bottom", "0"),
			n.css("border-bottom", "none"),
			n.css("padding", "6px"),
			n.css("border-bottom-left-radius", "0"),
			n.css("border-bottom-right-radius", "0"),
			n.wrap('<form class="uk-form uk-form-stack" action="#0"></form>'),
			n.after('<button type="button" onclick="' + i + "('" + r + '\', this)" class="uk-button uk-button-primary" style="margin-top:15px;"><i class="uk-icon-play"></i> Run</button>'),
			1 === s.length ? (s.unshift(""), s.push("")) : 2 === s.length && s.push(""),
			n.text(t(s[0])),
			s[2].trim() && (n.after('<pre id="post-' + r + '" style="font-size: 14px; margin-top: 0; border-top: 0; padding: 6px; border-top-left-radius: 0; border-top-right-radius: 0;"></pre>'), (a = $("#post-" + r)).text(t(s[2]))),
			n.after('<textarea id="textarea-' + r + '" onkeyup="adjustTextareaHeight(this)" class="uk-width-1-1 x-codearea" rows="10" style="overflow: scroll; border-top-left-radius: 0; border-top-right-radius: 0;' + (null === a ? "": "border-bottom-left-radius: 0; border-bottom-right-radius: 0;") + '"></textarea>'),
			$("#textarea-" + r).val(t(s[1])),
			adjustTextareaHeight($("#textarea-" + r).get(0))
		}
	} (), signinModal = null, tplComment = null, tplCommentReply = null, tplCommentInfo = null, tmp_collapse = 1;
	if ($(function() {
		$(".x-auto-content").each(function() {
			makeCollapsable(this, 300)
		})
	}), window.console || (window.console = {
		log: function(e) {}
	}), String.prototype.trim || (String.prototype.trim = function() {
		return this.replace(/^\s+|\s+$/g, "")
	}), !Number.prototype.toDateTime) {
		var replaces = {
			yyyy: function(e) {
				return e.getFullYear().toString()
			},
			yy: function(e) {
				return (e.getFullYear() % 100).toString()
			},
			MM: function(e) {
				var t = e.getMonth() + 1;
				return t < 10 ? "0" + t: t.toString()
			},
			M: function(e) {
				return (e.getMonth() + 1).toString()
			},
			dd: function(e) {
				var t = e.getDate();
				return t < 10 ? "0" + t: t.toString()
			},
			d: function(e) {
				return e.getDate().toString()
			},
			hh: function(e) {
				var t = e.getHours();
				return t < 10 ? "0" + t: t.toString()
			},
			h: function(e) {
				return e.getHours().toString()
			},
			mm: function(e) {
				var t = e.getMinutes();
				return t < 10 ? "0" + t: t.toString()
			},
			m: function(e) {
				return e.getMinutes().toString()
			},
			ss: function(e) {
				var t = e.getSeconds();
				return t < 10 ? "0" + t: t.toString()
			},
			s: function(e) {
				return e.getSeconds().toString()
			},
			a: function(e) {
				return e.getHours() < 12 ? "AM": "PM"
			}
		},
		token = /([a-zA-Z]+)/;
		Number.prototype.toDateTime = function(e) {
			for (var t = e || "yyyy-MM-dd hh:mm:ss",
			n = new Date(this), i = t.split(token), r = 0; r < i.length; r++) {
				var o = i[r];
				o && o in replaces && (i[r] = replaces[o](n))
			}
			return i.join("")
		},
		Number.prototype.toSmartDate = function() {
			return toSmartDate(this)
		}
	}
	$(function() {
		$.fn.extend({
			showFormError: function(e) {
				return this.each(function() {
					var t = $(this),
					n = t && t.find(".uk-alert-danger"),
					i = e && e.data;
					t.is("form") ? (t.find("input").removeClass("uk-form-danger"), t.find("select").removeClass("uk-form-danger"), t.find("textarea").removeClass("uk-form-danger"), 0 !== n.length ? e ? (n.text(e.message ? e.message: e.error ? e.error: e).removeClass("uk-hidden").show(), n.offset().top - 60 < $(window).scrollTop() && $("html,body").animate({
						scrollTop: n.offset().top - 60
					}), i && t.find("[name=" + i + "]").addClass("uk-form-danger")) : (n.addClass("uk-hidden").hide(), t.find(".uk-form-danger").removeClass("uk-form-danger")) : console.warn("Cannot find .uk-alert-danger element.")) : console.error("Cannot call showFormError() on non-form object.")
				})
			},
			showFormLoading: function(e) {
				return this.each(function() {
					var t = $(this),
					n = t && t.find("button[type=submit]"),
					i = t && t.find("button");
					$i = n && n.find("i"),
					iconClass = $i && $i.attr("class"),
					t.is("form") ? !iconClass || iconClass.indexOf("uk-icon") < 0 ? console.warn('Icon <i class="uk-icon-*>" not found.') : e ? (i.attr("disabled", "disabled"), $i && $i.addClass("uk-icon-spinner").addClass("uk-icon-spin")) : (i.removeAttr("disabled"), $i && $i.removeClass("uk-icon-spinner").removeClass("uk-icon-spin")) : console.error("Cannot call showFormLoading() on non-form object.")
				})
			},
			postJSON: function(e, t, n) {
				return 2 === arguments.length && (n = t, t = {}),
				this.each(function() {
					var i = $(this);
					i.showFormError(),
					i.showFormLoading(!0),
					_httpJSON("POST", e, t,
					function(e, t) {
						e && (i.showFormError(e), i.showFormLoading(!1)),
						n && n(e, t) && i.showFormLoading(!1)
					})
				})
			}
		})
	}), $(function() {
		var e = $('meta[property="x-nav"]').attr("content");
		e && e.trim() && $('#ul-navbar li a[href="' + e.trim() + '"]').parent().addClass("uk-active");
		var t = $(window),
		n = ($("body"), $("div.x-goto-top")),
		i = _.map($("img[data-src]").get(),
		function(e) {
			return $(e)
		}),
		r = function() {
			var e = t.scrollTop();
			if (e > 1600 ? n.show() : n.hide(), i.length > 0) {
				var r = t.height(),
				o = [];
				_.each(i,
				function(t, n) {
					t.offset().top - e < r && (t.attr("src", t.attr("data-src")), o.unshift(n))
				}),
				_.each(o,
				function(e) {
					i.splice(e, 1)
				})
			}
		};
		t.scroll(r),
		r(),
		n.click(function() {
			$("html, body").animate({
				scrollTop: 0
			},
			1e3)
		}),
		$(".x-smartdate").each(function() {
			var e = parseInt($(this).attr("date"));
			$(this).text(toSmartDate(e))
		}),
		$("pre>code").each(function(e, t) {
			var n = $(t),
			i = (n.attr("class") || "").split(" "),
			r = (_.find(i,
			function(e) {
				return e.indexOf("lang-nohightlight") >= 0
			}) || "").trim(),
			o = (_.find(i,
			function(e) {
				return e.indexOf("lang-!") >= 0
			}) || "").trim(),
			a = (_.find(i,
			function(e) {
				return e.indexOf("lang-?") >= 0
			}) || "").trim(),
			s = (_.find(i,
			function(e) {
				return e.indexOf("lang-x-") >= 0
			}) || "").trim();
			if (n.hasClass("lang-ascii")) n.css("font-family", '"Courier New",Consolas,monospace').parent("pre").css("font-size", "12px").css("line-height", "12px").css("border", "none").css("background-color", "transparent");
			else if (o || a) n.parent().replaceWith('<div class="uk-alert ' + (o ? "uk-alert-danger": "") + '"><i class="uk-icon-' + (o ? "warning": "info-circle") + '"></i> ' + encodeHtml(n.text()) + "</div>");
			else if (s) {
				var l = "run_" + s.substring(7);
				initRunCode(n.parent(), l)
			} else r || hljs.highlightBlock(t)
		})
	});
	var isDesktop = function() {
		var e = navigator.userAgent.toLowerCase();
		return e.indexOf("windows nt") >= 0 || e.indexOf("macintosh") >= 0
	} ();