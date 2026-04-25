import { generateAnswersWithOpenAiApiCompat } from './openai-api.mjs'
import { getModelValue } from '../../utils/model-name-convert.mjs'

/**
 * @param {Browser.Runtime.Port} port
 * @param {string} question
 * @param {Session} session
 * @param {string} apiKey
 */
export async function generateAnswersWithDeepSeekApi(port, question, session, apiKey) {
  const baseUrl = 'https://api.deepseek.com'
  const modelValue = getModelValue(session)

  // DeepSeek 官方 API 的思考模式默认开启，但部分调用方（如 selection-tools 中的 Explain）
  // 是一次性的轻量解释/翻译任务，不需要思考过程，因此通过 session.disableThinking 显式关闭。
  // 仅对支持思考模式开关的 deepseek-v4 生效
  // 参考: https://api-docs.deepseek.com/guides/thinking_mode
  const extraBody = {}
  const thinkingToggleableModels = new Set(['deepseek-v4-flash', 'deepseek-v4-pro'])
  if (thinkingToggleableModels.has(modelValue) && session?.disableThinking) {
    extraBody.thinking = { type: 'disabled' }
  }

  return generateAnswersWithOpenAiApiCompat(baseUrl, port, question, session, apiKey, extraBody)
}
