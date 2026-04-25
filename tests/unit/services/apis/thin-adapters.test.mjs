import assert from 'node:assert/strict'
import { beforeEach, test } from 'node:test'
import { createFakePort } from '../../helpers/port.mjs'
import { createMockSseResponse } from '../../helpers/sse-response.mjs'

import { generateAnswersWithAimlApi } from '../../../../src/services/apis/aiml-api.mjs'
import { generateAnswersWithDeepSeekApi } from '../../../../src/services/apis/deepseek-api.mjs'
import { generateAnswersWithMoonshotCompletionApi } from '../../../../src/services/apis/moonshot-api.mjs'
import { generateAnswersWithOpenRouterApi } from '../../../../src/services/apis/openrouter-api.mjs'
import { generateAnswersWithChatGLMApi } from '../../../../src/services/apis/chatglm-api.mjs'

const setStorage = (values) => {
  globalThis.__TEST_BROWSER_SHIM__.replaceStorage(values)
}

beforeEach(() => {
  globalThis.__TEST_BROWSER_SHIM__.clearStorage()
})

const commonStorage = {
  maxConversationContextLength: 3,
  maxResponseTokenLength: 256,
  temperature: 0.5,
}

const makeSession = () => ({
  modelName: 'chatgptApi4oMini',
  conversationRecords: [],
  isRetry: false,
})

const sseChunks = ['data: {"choices":[{"delta":{"content":"OK"},"finish_reason":"stop"}]}\n\n']

const adapters = [
  {
    name: 'aiml-api',
    fn: (port, q, session) => generateAnswersWithAimlApi(port, q, session, 'aiml-key'),
    expectedBaseUrl: 'https://api.aimlapi.com/v1',
    expectedApiKey: 'aiml-key',
    storage: commonStorage,
  },
  {
    name: 'deepseek-api',
    fn: (port, q, session) => generateAnswersWithDeepSeekApi(port, q, session, 'ds-key'),
    expectedBaseUrl: 'https://api.deepseek.com',
    expectedApiKey: 'ds-key',
    storage: commonStorage,
  },
  {
    name: 'moonshot-api',
    fn: (port, q, session) => generateAnswersWithMoonshotCompletionApi(port, q, session, 'ms-key'),
    expectedBaseUrl: 'https://api.moonshot.cn/v1',
    expectedApiKey: 'ms-key',
    storage: commonStorage,
  },
  {
    name: 'openrouter-api',
    fn: (port, q, session) => generateAnswersWithOpenRouterApi(port, q, session, 'or-key'),
    expectedBaseUrl: 'https://openrouter.ai/api/v1',
    expectedApiKey: 'or-key',
    storage: commonStorage,
  },
  {
    name: 'chatglm-api',
    fn: (port, q, session) => generateAnswersWithChatGLMApi(port, q, session),
    expectedBaseUrl: 'https://open.bigmodel.cn/api/paas/v4',
    expectedApiKey: 'glm-key',
    storage: { ...commonStorage, chatglmApiKey: 'glm-key' },
  },
]

for (const adapter of adapters) {
  test(`${adapter.name}: passes correct base URL and API key`, async (t) => {
    t.mock.method(console, 'debug', () => {})
    setStorage(adapter.storage)

    const session = makeSession()
    const port = createFakePort()

    let capturedInput, capturedInit
    t.mock.method(globalThis, 'fetch', async (input, init) => {
      capturedInput = input
      capturedInit = init
      return createMockSseResponse(sseChunks)
    })

    await adapter.fn(port, 'Q', session)

    assert.equal(capturedInput, `${adapter.expectedBaseUrl}/chat/completions`)
    // Verify API key reaches the Authorization header
    assert.equal(capturedInit.headers.Authorization, `Bearer ${adapter.expectedApiKey}`)
  })

  test(`${adapter.name}: delegates to compat layer and produces output`, async (t) => {
    t.mock.method(console, 'debug', () => {})
    setStorage(adapter.storage)

    const session = makeSession()
    const port = createFakePort()

    t.mock.method(globalThis, 'fetch', async () => createMockSseResponse(sseChunks))

    await adapter.fn(port, 'Q', session)

    assert.equal(
      port.postedMessages.some((m) => m.done === true && m.session === session),
      true,
    )
    assert.deepEqual(session.conversationRecords.at(-1), {
      question: 'Q',
      answer: 'OK',
    })
  })
}

test('chatglm-api: reads chatglmApiKey from config', async (t) => {
  t.mock.method(console, 'debug', () => {})
  setStorage({ ...commonStorage, chatglmApiKey: 'glm-secret' })

  const session = makeSession()
  const port = createFakePort()

  let capturedInit
  t.mock.method(globalThis, 'fetch', async (_input, init) => {
    capturedInit = init
    return createMockSseResponse(sseChunks)
  })

  await generateAnswersWithChatGLMApi(port, 'Q', session)

  assert.equal(capturedInit.headers.Authorization, 'Bearer glm-secret')
})

const captureDeepSeekBody = async (t, sessionOverrides) => {
  t.mock.method(console, 'debug', () => {})
  setStorage(commonStorage)

  const session = { ...makeSession(), ...sessionOverrides }
  const port = createFakePort()

  let capturedInit
  t.mock.method(globalThis, 'fetch', async (_input, init) => {
    capturedInit = init
    return createMockSseResponse(sseChunks)
  })

  await generateAnswersWithDeepSeekApi(port, 'Q', session, 'ds-key')
  return JSON.parse(capturedInit.body)
}

// 通过 customApiModelKeys 这种「Always Custom」分组构造可被 getModelValue 直接解析为
// 想要 modelValue 的 apiMode，避免依赖 Models 配置中具体的 deepseek 模型 key 映射。
const apiModeForCustomValue = (customName) => ({
  groupName: 'customApiModelKeys',
  customName,
})

test('deepseek-api: deepseek-v4-flash keeps thinking by default (no override)', async (t) => {
  const body = await captureDeepSeekBody(t, {
    apiMode: apiModeForCustomValue('deepseek-v4-flash'),
  })
  assert.equal(body.thinking, undefined)
})

test('deepseek-api: deepseek-v4-flash disables thinking when session.disableThinking is true', async (t) => {
  const body = await captureDeepSeekBody(t, {
    apiMode: apiModeForCustomValue('deepseek-v4-flash'),
    disableThinking: true,
  })
  assert.deepEqual(body.thinking, { type: 'disabled' })
})

test('deepseek-api: deepseek-v4-pro disables thinking when session.disableThinking is true', async (t) => {
  const body = await captureDeepSeekBody(t, {
    apiMode: apiModeForCustomValue('deepseek-v4-pro'),
    disableThinking: true,
  })
  assert.deepEqual(body.thinking, { type: 'disabled' })
})

test('deepseek-api: non-toggleable models never send thinking override even if disableThinking set', async (t) => {
  const body = await captureDeepSeekBody(t, {
    modelName: 'deepseek_reasoner',
    disableThinking: true,
  })
  assert.equal(body.thinking, undefined)
})
