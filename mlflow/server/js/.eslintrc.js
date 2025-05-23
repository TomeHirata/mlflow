// eslint-disable-next-line import/no-extraneous-dependencies
const restrictedGlobals = require('confusing-browser-globals');

module.exports = {
  extends: ['prettier'],
  plugins: ['prettier', 'no-only-tests', 'formatjs', 'react', 'import', 'jsx-a11y', 'react-hooks'],
  parser: '@babel/eslint-parser',
  parserOptions: {
    sourceType: 'module',
    ecmaVersion: 7,
    ecmaFeatures: {
      jsx: true,
    },
    babelOptions: {
      presets: ['@babel/preset-react'],
    },
  },
  env: {
    es6: true,
    browser: true,
    jest: true,
  },
  globals: {
    GridStackUI: true,
    jQuery: true,
    MG: true,
  },
  settings: {
    'import/resolver': {
      webpack: {
        config: {
          resolve: {
            // eslint-disable-next-line no-process-env
            modules: ['node_modules', process.env.NODE_MODULES_PATH],
            extensions: ['.js', '.jsx', '.ts', '.tsx'],
          },
        },
      },
    },
    react: {
      version: 'detect',
    },
  },
  rules: {
    // http://eslint.org/docs/rules/
    'array-callback-return': 'warn',
    'default-case': ['warn', { commentPattern: '^no default$' }],
    'dot-location': ['warn', 'property'],
    eqeqeq: ['warn', 'smart'],
    'new-parens': 'warn',
    'no-array-constructor': 'warn',
    'no-caller': 'warn',
    'no-cond-assign': ['warn', 'except-parens'],
    'no-const-assign': 'warn',
    'no-control-regex': 'warn',
    'no-delete-var': 'warn',
    'no-dupe-args': 'warn',
    'no-dupe-class-members': 'warn',
    'no-dupe-keys': 'warn',
    'no-duplicate-case': 'warn',
    'no-empty-character-class': 'warn',
    'no-empty-pattern': 'warn',
    'no-eval': 'warn',
    'no-ex-assign': 'warn',
    'no-extend-native': 'warn',
    'no-extra-bind': 'warn',
    'no-extra-label': 'warn',
    'no-fallthrough': 'warn',
    'no-func-assign': 'warn',
    'no-implied-eval': 'warn',
    'no-invalid-regexp': 'warn',
    'no-iterator': 'warn',
    'no-label-var': 'warn',
    'no-labels': ['warn', { allowLoop: true, allowSwitch: false }],
    'no-lone-blocks': 'warn',
    'no-loop-func': 'warn',
    'no-mixed-operators': [
      'warn',
      {
        groups: [
          ['&', '|', '^', '~', '<<', '>>', '>>>'],
          ['==', '!=', '===', '!==', '>', '>=', '<', '<='],
          ['&&', '||'],
          ['in', 'instanceof'],
        ],
        allowSamePrecedence: false,
      },
    ],
    'no-multi-str': 'warn',
    'no-global-assign': 'warn',
    'no-unsafe-negation': 'warn',
    'no-new-func': 'warn',
    'no-new-object': 'warn',
    'no-new-symbol': 'warn',
    'no-new-wrappers': 'warn',
    'no-obj-calls': 'warn',
    'no-octal': 'warn',
    'no-octal-escape': 'warn',
    'no-redeclare': 'warn',
    'no-regex-spaces': 'warn',
    'no-restricted-syntax': ['warn', 'WithStatement'],
    'no-script-url': 'warn',
    'no-self-assign': 'warn',
    'no-self-compare': 'warn',
    'no-sequences': 'warn',
    'no-shadow-restricted-names': 'warn',
    'no-sparse-arrays': 'warn',
    'no-template-curly-in-string': 'warn',
    'no-this-before-super': 'warn',
    'no-throw-literal': 'warn',
    'no-undef': 'error',
    'no-restricted-globals': ['error'].concat(restrictedGlobals),
    'no-unreachable': 'warn',
    'no-unused-expressions': [
      'error',
      {
        allowShortCircuit: true,
        allowTernary: true,
        allowTaggedTemplates: true,
      },
    ],
    'no-unused-labels': 'warn',
    'no-unused-vars': [
      'warn',
      {
        args: 'none',
        ignoreRestSiblings: true,
      },
    ],
    'no-use-before-define': [
      'warn',
      {
        functions: false,
        classes: false,
        variables: false,
      },
    ],
    'no-useless-computed-key': 'warn',
    'no-useless-concat': 'warn',
    'no-useless-constructor': 'warn',
    'no-useless-escape': 'warn',
    'no-useless-rename': [
      'warn',
      {
        ignoreDestructuring: false,
        ignoreImport: false,
        ignoreExport: false,
      },
    ],
    'no-with': 'warn',
    'no-whitespace-before-property': 'warn',
    'react-hooks/exhaustive-deps': 'warn',
    'require-yield': 'warn',
    'rest-spread-spacing': ['warn', 'never'],
    strict: ['warn', 'never'],
    'unicode-bom': ['warn', 'never'],
    'use-isnan': 'warn',
    'valid-typeof': 'warn',
    'no-restricted-properties': [
      'error',
      {
        object: 'require',
        property: 'ensure',
        message:
          'Please use import() instead. More info: https://facebook.github.io/create-react-app/docs/code-splitting',
      },
      {
        object: 'System',
        property: 'import',
        message:
          'Please use import() instead. More info: https://facebook.github.io/create-react-app/docs/code-splitting',
      },
    ],
    'getter-return': 'warn',

    // https://github.com/benmosher/eslint-plugin-import/tree/master/docs/rules
    // 'import/first': 'error',
    'import/no-amd': 'error',
    'import/no-anonymous-default-export': 'warn',
    // 'import/no-webpack-loader-syntax': 'error',

    // https://github.com/yannickcr/eslint-plugin-react/tree/master/docs/rules
    'react/forbid-foreign-prop-types': ['warn', { allowInPropTypes: true }],
    'react/jsx-no-comment-textnodes': 'warn',
    'react/jsx-no-duplicate-props': 'warn',
    'react/jsx-no-target-blank': 'warn',
    'react/jsx-no-undef': 'error',
    'react/jsx-pascal-case': [
      'warn',
      {
        allowAllCaps: true,
        ignore: [],
      },
    ],
    'react/no-danger-with-children': 'warn',
    'react/no-direct-mutation-state': 'warn',
    'react/no-is-mounted': 'warn',
    'react/no-typos': 'error',
    'react/require-render-return': 'error',
    'react/style-prop-object': 'warn',

    // https://github.com/evcohen/eslint-plugin-jsx-a11y/tree/master/docs/rules
    'jsx-a11y/alt-text': 'warn',
    'jsx-a11y/anchor-has-content': 'warn',
    'jsx-a11y/anchor-is-valid': [
      'warn',
      {
        aspects: ['noHref', 'invalidHref'],
      },
    ],
    'jsx-a11y/aria-activedescendant-has-tabindex': 'warn',
    'jsx-a11y/aria-props': 'warn',
    'jsx-a11y/aria-proptypes': 'warn',
    'jsx-a11y/aria-role': ['warn', { ignoreNonDOM: true }],
    'jsx-a11y/aria-unsupported-elements': 'warn',
    'jsx-a11y/heading-has-content': 'warn',
    'jsx-a11y/iframe-has-title': 'warn',
    'jsx-a11y/img-redundant-alt': 'warn',
    'jsx-a11y/no-access-key': 'warn',
    'jsx-a11y/no-distracting-elements': 'warn',
    'jsx-a11y/no-redundant-roles': 'warn',
    'jsx-a11y/role-has-required-aria-props': 'warn',
    'jsx-a11y/role-supports-aria-props': 'warn',
    'jsx-a11y/scope': 'warn',

    // https://github.com/facebook/react/tree/main/packages/eslint-plugin-react-hooks
    'react-hooks/rules-of-hooks': 'error',

    'accessor-pairs': 2,
    'array-bracket-spacing': 2,
    'arrow-body-style': 'off',
    'arrow-parens': 'off',
    'arrow-spacing': 2,
    'block-scoped-var': 2,
    'block-spacing': 2,
    'brace-style': [2, '1tbs', { allowSingleLine: false }],
    'callback-return': 2,
    camelcase: 'off',
    'class-methods-use-this': 0,
    'comma-dangle': [2, 'always-multiline'],
    'comma-spacing': 2,
    'comma-style': 2,
    'computed-property-spacing': 2,
    'consistent-return': 'off',
    'consistent-this': 0,
    'constructor-super': 2,
    curly: 2,
    'dot-notation': 'off',
    'eol-last': 2,
    'func-call-spacing': 2,
    'func-names': 2,
    'func-style': 0,
    'generator-star-spacing': 2,
    'global-require': 2,
    'guard-for-in': 2,
    'handle-callback-err': 2,
    'id-blacklist': 2,
    'id-length': 0,
    'id-match': 2,
    'import/default': 0,
    'import/export': 2,
    'import/extensions': [
      'error',
      'ignorePackages',
      {
        js: 'never',
        jsx: 'never',
        ts: 'never',
        tsx: 'never',
      },
    ],
    'import/first': 0, // FEINF-2715 - make this 'error' and fix
    'import/max-dependencies': 0,
    'import/named': 0,
    'import/namespace': 2,
    'import/newline-after-import': 2,
    'import/no-absolute-path': 0,
    'import/no-commonjs': 0,
    'import/no-deprecated': 1,
    'import/no-duplicates': 2,
    'import/no-dynamic-require': 0,
    'import/no-extraneous-dependencies': [2, { devDependencies: true }],
    'import/no-internal-modules': 0,
    'import/no-mutable-exports': 2,
    'import/no-named-as-default': 0,
    'import/no-named-as-default-member': 2,
    'import/no-namespace': 2,
    'import/no-nodejs-modules': 2,
    'import/no-restricted-paths': 2,
    'import/no-unassigned-import': 0,
    'import/no-unresolved': 0,
    'import/no-webpack-loader-syntax': 0,
    'import/order': 0,
    'import/prefer-default-export': 0,
    'import/unambiguous': 0,
    indent: 0,
    'indent-legacy': 0,
    'init-declarations': 0,
    'jsx-quotes': 'off',
    'key-spacing': 2,
    'keyword-spacing': 2,
    'linebreak-style': 2,
    'lines-around-comment': 0,
    'max-depth': [2, 4],
    'max-lines': [2, 1500],
    'max-nested-callbacks': 2,
    'max-params': [2, 12],
    'max-statements': 0,
    'max-statements-per-line': 2,
    'multiline-ternary': 0,
    'new-cap': 0,
    'newline-after-var': 0,
    'newline-before-return': 0,
    'newline-per-chained-call': 'off',
    'no-alert': 2,
    'no-bitwise': 2,
    'no-case-declarations': 'off',
    'no-catch-shadow': 'off',
    'no-class-assign': 2,
    'no-confusing-arrow': 0,
    'no-console': [2, { allow: ['warn', 'error'] }],
    'no-constant-condition': 2,
    'no-continue': 0,
    'no-debugger': 2,
    'no-div-regex': 2,
    'no-else-return': 'off',
    'no-empty': 2,
    'no-eq-null': 2,
    'no-extra-boolean-cast': 2,
    'no-extra-parens': 0,
    'no-extra-semi': 2,
    'no-floating-decimal': 2,
    'no-implicit-coercion': 0,
    'no-implicit-globals': 2,
    'no-inline-comments': 0,
    'no-inner-declarations': 2,
    'no-invalid-this': 0,
    'no-irregular-whitespace': 2,
    'no-lonely-if': 2,
    'no-magic-numbers': 0,
    'no-mixed-requires': 0,
    'no-mixed-spaces-and-tabs': 2,
    'no-multi-spaces': ['error', { ignoreEOLComments: true }],
    'no-multiple-empty-lines': 2,
    'no-negated-condition': 0,
    'no-nested-ternary': 'off',
    'no-new': 0,
    'no-new-require': 2,
    'no-param-reassign': 2,
    'no-path-concat': 2,
    'no-plusplus': 0,
    'no-process-env': 0,
    'no-process-exit': 2,
    'no-proto': 2,
    'no-prototype-builtins': 0,
    'no-restricted-imports': [
      'error',
      {
        paths: [
          {
            name: 'emotion',
            message:
              // eslint-disable-next-line max-len
              "Importing emotion is obsolete - please use css={...} prop in JSX elements now. For class names, you can import { ClassNames } from '@emotion/react' package.",
          },
        ],
        patterns: [
          {
            group: ['react-router*'],
            message:
              // eslint-disable-next-line max-len
              'Please do not import from react-router libraries directly and use `src/common/utils/RoutingUtils` module instead.',
          },
        ],
      },
    ],
    'no-restricted-modules': 2,
    'no-return-assign': 0,
    'no-shadow': 2,
    'no-sync': 2,
    'no-tabs': 2,
    'no-ternary': 0,
    'no-trailing-spaces': 2,
    'no-undef-init': 'off',
    'no-undefined': 0,
    'no-underscore-dangle': 0,
    'no-unexpected-multiline': 2,
    'no-unmodified-loop-condition': 2,
    'no-unneeded-ternary': 2,
    'no-unsafe-finally': 2,
    'no-useless-call': 2,
    'no-var': 2,
    'no-void': 2,
    'no-warning-comments': 0,
    'object-curly-newline': 0,
    'object-curly-spacing': 'off',
    'object-property-newline': 0,
    'object-shorthand': [2, 'methods'],
    'one-var': 0,
    'one-var-declaration-per-line': 2,
    'operator-assignment': 2,
    'operator-linebreak': 0,
    'padded-blocks': 0,
    'prefer-arrow-callback': 0,
    'prefer-const': 2,
    'prefer-reflect': 0,
    'prefer-rest-params': 2,
    'prefer-spread': 2,
    'prefer-template': 0,
    'quote-props': 0,
    quotes: 'off',
    radix: 2,
    'react/display-name': 0,
    'react/forbid-component-props': 0,
    'react/forbid-prop-types': 0,
    'react/jsx-boolean-value': 2,
    'react/jsx-closing-bracket-location': 0,
    'react/jsx-curly-spacing': 2,
    'react/jsx-equals-spacing': 2,
    'react/jsx-filename-extension': 0,
    'react/jsx-first-prop-new-line': 0,
    'react/jsx-handler-names': 0,
    'react/jsx-indent': 0,
    'react/jsx-indent-props': 0,
    'react/jsx-key': 0,
    'react/jsx-max-props-per-line': 0,
    'react/jsx-no-bind': [0, { ignoreRefs: true }],
    'react/jsx-no-literals': 0,
    'react/jsx-sort-props': 0,
    'react/jsx-space-before-closing': 0,
    'react/jsx-uses-react': 2,
    'react/jsx-uses-vars': 2,
    'react/jsx-wrap-multilines': 0,
    'react/no-children-prop': 0,
    'react/no-danger': 2,
    'react/no-did-mount-set-state': 2,
    'react/no-did-update-set-state': 2,
    'react/no-find-dom-node': 0,
    'react/no-multi-comp': 0,
    'react/no-render-return-value': 0,
    'react/no-set-state': 0,
    'react/no-string-refs': 0,
    'react/no-unescaped-entities': 0,
    'react/no-unknown-property': ['error', { ignore: ['css'] }],
    'react/no-unused-prop-types': 0,
    'react/prefer-es6-class': 0,
    'react/prefer-stateless-function': 0,
    'react/prop-types': 2,
    'react/react-in-jsx-scope': 2,
    'react/require-optimization': 0,
    'react/self-closing-comp': 0,
    'react/sort-comp': 0,
    'react/sort-prop-types': 0,
    'require-jsdoc': 0,
    semi: 2,
    'semi-spacing': 2,
    'sort-imports': 0,
    'sort-keys': 0,
    'sort-vars': 0,
    'space-before-blocks': 2,
    'space-before-function-paren': [
      2,
      {
        anonymous: 'always',
        named: 'never',
        asyncArrow: 'always',
      },
    ],
    'space-in-parens': 2,
    'space-infix-ops': 2,
    'space-unary-ops': 0,
    'spaced-comment': [2, 'always', { exceptions: ['/'], markers: ['/'] }],
    'symbol-description': 2,
    'template-curly-spacing': 2,
    'valid-jsdoc': 0,
    'vars-on-top': 0,
    'wrap-iife': 2,
    'wrap-regex': 0,
    'yield-star-spacing': 2,
    yoda: 2,
    'function-paren-newline': 'off',
    complexity: 'off',
    'no-multi-assign': 'off',
    'no-useless-return': 'off',
    'prefer-destructuring': 'off',

    // Rules specific to "no-only-tests" plugin:
    'no-only-tests/no-only-tests': 2,
  },
  overrides: [
    {
      // We're enabling '@typescript-eslint/parser' and its rules only for the *.ts(x) files
      files: ['*.ts', '*.tsx'],
      extends: ['prettier', 'plugin:jsx-a11y/recommended', 'plugin:@typescript-eslint/recommended'],
      plugins: ['prettier', '@typescript-eslint', '@emotion'],
      parser: '@typescript-eslint/parser',
      parserOptions: {
        ecmaVersion: 2018,
        sourceType: 'module',
        ecmaFeatures: {
          jsx: true,
        },

        // typescript-eslint specific options
        warnOnUnsupportedTypeScriptVersion: true,
      },

      rules: {
        // react-app original rules

        // TypeScript's `noFallthroughCasesInSwitch` option is more robust (#6906)
        'default-case': 'off',
        // 'tsc' already handles this (https://github.com/typescript-eslint/typescript-eslint/issues/291)
        'no-dupe-class-members': 'off',
        // 'tsc' already handles this (https://github.com/typescript-eslint/typescript-eslint/issues/477)
        'no-undef': 'off',

        // Add TypeScript specific rules (and turn off ESLint equivalents)
        '@typescript-eslint/consistent-type-assertions': 'warn',
        'no-array-constructor': 'off',
        '@typescript-eslint/no-array-constructor': 'warn',
        'no-redeclare': 'off',
        '@typescript-eslint/no-redeclare': 'warn',
        'no-use-before-define': 'off',
        '@typescript-eslint/no-use-before-define': [
          'warn',
          {
            functions: false,
            classes: false,
            variables: false,
            typedefs: false,
          },
        ],
        'no-unused-expressions': 'off',
        '@typescript-eslint/no-unused-expressions': [
          'error',
          {
            allowShortCircuit: true,
            allowTernary: true,
            allowTaggedTemplates: true,
          },
        ],
        'no-unused-vars': 'off',
        // '@typescript-eslint/no-unused-vars': [
        //   'warn',
        //   {
        //     args: 'none',
        //     ignoreRestSiblings: true,
        //   },
        // ],
        'no-useless-constructor': 'off',
        '@typescript-eslint/no-useless-constructor': 'warn',
        // end react-app original rules

        // Turning off temporarily until TS migration is complete
        'import/first': 0,
        'import/extensions': 0,
        'import/newline-after-import': 0,
        'import/no-duplicates': 0,
        'import/namespace': 0,
        '@typescript-eslint/no-unused-vars': 0,
        'react/prop-types': 0,
        'max-lines': 0,
        'jsx-a11y/click-events-have-key-events': 0,
        'jsx-a11y/no-static-element-interactions': 0,
        'jsx-a11y/interactive-supports-focus': 0,
        'jsx-a11y/label-has-associated-control': 0,
        'jsx-a11y/no-noninteractive-element-interactions': 0,
        'jsx-a11y/no-autofocus': 0,

        // Do not require functions (especially react components) to have explicit returns
        '@typescript-eslint/explicit-function-return-type': 'off',
        // Do not require to type every import from a JS file to speed up development
        '@typescript-eslint/no-explicit-any': 'off',
        'no-empty-function': 'off',
        '@typescript-eslint/no-empty-function': ['error', { allow: ['arrowFunctions', 'methods'] }],
        // Many API fields and generated types use camelcase
        '@typescript-eslint/naming-convention': 'off',

        // TODO(thielium): This should be re-enabled (REDASH-796)
        '@typescript-eslint/explicit-module-boundary-types': 'off',

        // ts-migrate introduces a lot of ts-expect-error. turning into warning until we finalize the migration
        '@typescript-eslint/ban-ts-comment': 'warn',

        'no-shadow': 'off',
        '@typescript-eslint/no-shadow': 'off',

        // Please leave this rule as the last item in the array, as it's quite large
        '@typescript-eslint/ban-types': [
          'error',
          {
            types: {
              Function: {
                message:
                  'The `Function` type accepts any function-like value. It provides no type safety when calling the function, which can be a common source of bugs.\nIt also accepts things like class declarations, which will throw at runtime as they will not be called with `new`.\nIf you are expecting the function to accept certain arguments, you should explicitly define the function shape.',
                fixWith: '(...args: unknown[]) => unknown',
              },
              '{}': {
                message:
                  '`{}` actually means "any non-nullish value".\n- If you want a type meaning "any object", you probably want `Record<string, unknown>` instead.\n- If you want a type meaning "any value", you probably want `unknown` instead.\n- If you want a type meaning "empty object", you probably want `Record<string, never>` instead.',
                fixWith: 'Record<string, never>',
              },
              Object: {
                message:
                  '`Object` actually means "any non-nullish value".\n- If you want a type meaning "any object", you probably want `Record<string, unknown>` instead.\n- If you want a type meaning "any value", you probably want `unknown` instead.\n- If you want a type meaning "empty object", you probably want `Record<string, never>` instead.',
                fixWith: 'Record<string, unknown>',
              },
              object: {
                message:
                  "Don't use `object` as a type. The `object` type is currently hard to use ([see this issue](https://github.com/microsoft/TypeScript/issues/21732)).\nConsider using `Record<string, unknown>` instead, as it allows you to more easily inspect and use the keys.",
                fixWith: 'Record<string, unknown>',
              },
            },
          },
        ],
        // By using "auto" JSX runtime in TS, we have react automatically injected and
        // adding "React" manually results in TS(6133) error
        'react/react-in-jsx-scope': 'off',

        // '@typescript-eslint/no-unused-vars': ['error', { varsIgnorePattern: '^oss_' }],
      },
    },
    {
      files: ['*.test.js', '*-test.js', '*-test.jsx', 'test/**'],
      plugins: ['jest', 'chai-expect', 'chai-friendly'],
      globals: {
        sinon: true,
        chai: true,
        expect: true,
        assert: true,
      },
      rules: {
        'func-names': 0,
        'max-lines': 0,
        'chai-expect/missing-assertion': 2,
        'no-unused-expressions': 0,
        'chai-friendly/no-unused-expressions': 2,
        'testing-library/no-debugging-utils': 'error',
        'testing-library/no-dom-import': 'error',
        'testing-library/await-async-utils': 'error',
      },
    },
    {
      files: ['src/**/*.stories.*'],
      rules: {
        'import/no-anonymous-default-export': 'off',
      },
    },
  ],
};
