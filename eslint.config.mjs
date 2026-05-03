import js from "@eslint/js";
import html from "eslint-plugin-html";

export default [
  {
    files: ["**/*.html"],
    plugins: { html },
    rules: {
      ...js.configs.recommended.rules,
      // Allow browser globals (window, document, localStorage, etc.)
      "no-undef": "off",
      // Allow unused vars in HTML scripts (many are called from onclick etc.)
      "no-unused-vars": "off",
      // Empty catch {} is used intentionally (mkdir exists, resilient JSON parse)
      "no-empty": ["warn", { allowEmptyCatch: true }],
      // Unnecessary escapes are style issues, not bugs
      "no-useless-escape": "warn",
      // These catch the class of bugs we care about
      "no-unreachable": "error",
    },
    languageOptions: {
      ecmaVersion: 2022,
      sourceType: "script",
      globals: {
        window: "readonly", document: "readonly", console: "readonly",
        fetch: "readonly", localStorage: "readonly", sessionStorage: "readonly",
        setTimeout: "readonly", clearTimeout: "readonly", setInterval: "readonly",
        clearInterval: "readonly", AbortController: "readonly", TextDecoder: "readonly",
        URL: "readonly", URLSearchParams: "readonly", FormData: "readonly",
        alert: "readonly", confirm: "readonly", marked: "readonly",
        CodeMirror: "readonly", Pyodide: "readonly", loadPyodide: "readonly",
      },
    },
  },
];
