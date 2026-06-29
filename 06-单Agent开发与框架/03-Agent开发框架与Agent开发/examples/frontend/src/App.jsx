import { useState, useRef, useEffect } from 'react'
import './App.css'
import SessionSidebar from './components/SessionSidebar'
import ContextPanel from './components/ContextPanel'
import ToolsPanel from './components/ToolsPanel'
import SkillsPanel from './components/SkillsPanel'

// API 地址（可通过环境变量配置）
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'
const JAVA_API_URL = 'http://localhost:8080'

function getApiUrl(backend) {
  return backend === 'java' ? JAVA_API_URL : API_URL
}

function App() {
  // 对话状态
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [currentThreadId, setCurrentThreadId] = useState('default')
  const [backend, setBackend] = useState('python')
  const messagesEndRef = useRef(null)

  // 右侧面板状态
  const [rightPanel, setRightPanel] = useState('context') // 'context' | 'tools' | 'skills'
  const [showRightPanel, setShowRightPanel] = useState(true)

  // 自动滚动到底部
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // 切换线程时清空消息
  useEffect(() => {
    setMessages([])
  }, [currentThreadId])

  // 发送消息
  const sendMessage = async () => {
    if (!input.trim() || loading) return

    const userMessage = input.trim()
    setInput('')
    setMessages(prev => [...prev, { role: 'user', content: userMessage }])
    setLoading(true)

    // 添加一个空的 assistant 消息用于流式填充
    setMessages(prev => [...prev, { role: 'assistant', content: '', toolCalls: [] }])

    try {
      const response = await fetch(`${getApiUrl(backend)}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          message: userMessage,
          thread_id: currentThreadId,  // 使用当前线程ID
          mode: 'react'
        }),
      })

      const reader = response.body.getReader()
      const decoder = new TextDecoder()

      let fullContent = ''
      let toolCalls = []

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        const text = decoder.decode(value)
        const lines = text.split('\n').filter(line => line.startsWith('data:'))

        for (const line of lines) {
          const dataStr = line.slice(5).trim()
          if (!dataStr) continue

          try {
            const event = JSON.parse(dataStr)

            if (event.type === 'token') {
              fullContent += event.content
              setMessages(prev => {
                const updated = [...prev]
                updated[updated.length - 1] = {
                  ...updated[updated.length - 1],
                  content: fullContent
                }
                return updated
              })
            } else if (event.type === 'tool_call') {
              toolCalls.push({ name: event.name, args: event.args, status: 'calling' })
              setMessages(prev => {
                const updated = [...prev]
                updated[updated.length - 1] = {
                  ...updated[updated.length - 1],
                  toolCalls: [...toolCalls]
                }
                return updated
              })
            } else if (event.type === 'tool_result') {
              const idx = toolCalls.findIndex(t => t.name === event.name && t.status === 'calling')
              if (idx >= 0) {
                toolCalls[idx] = { ...toolCalls[idx], result: event.result, status: 'done' }
                setMessages(prev => {
                  const updated = [...prev]
                  updated[updated.length - 1] = {
                    ...updated[updated.length - 1],
                    toolCalls: [...toolCalls]
                  }
                  return updated
                })
              }
            }
          } catch (e) {
            console.error('Parse error:', e)
          }
        }
      }
    } catch (err) {
      console.error('Chat error:', err)
      setMessages(prev => {
        const updated = [...prev]
        updated[updated.length - 1] = {
          ...updated[updated.length - 1],
          content: '抱歉，发生了错误，请重试。'
        }
        return updated
      })
    } finally {
      setLoading(false)
    }
  }

  // 渲染右侧面板
  const renderRightPanel = () => {
    if (!showRightPanel) return null
    
    switch (rightPanel) {
      case 'context':
        return <ContextPanel currentThreadId={currentThreadId} getApiUrl={getApiUrl} backend={backend} />
      case 'tools':
        return <ToolsPanel getApiUrl={getApiUrl} backend={backend} />
      case 'skills':
        return <SkillsPanel getApiUrl={getApiUrl} backend={backend} />
      default:
        return null
    }
  }

  return (
    <div className="app">
      {/* 左侧：会话列表 */}
      <SessionSidebar 
        currentThreadId={currentThreadId}
        onSelectThread={setCurrentThreadId}
        getApiUrl={getApiUrl}
        backend={backend}
      />

      {/* 中间：对话区域 */}
      <div className="main-content">
        <header className="header">
          <h1>Agent App</h1>
          <p className="subtitle">LangGraph / Spring AI 智能体演示</p>
          
          <div className="header-controls">
            {/* 后端切换 */}
            <div className="backend-selector">
              <button 
                className={backend === 'python' ? 'active' : ''}
                onClick={() => setBackend('python')}
              >
                Python
              </button>
              <button 
                className={backend === 'java' ? 'active' : ''}
                onClick={() => setBackend('java')}
              >
                Java
              </button>
            </div>

            {/* 右侧面板切换 */}
            <div className="panel-toggle">
              <button 
                className={showRightPanel && rightPanel === 'context' ? 'active' : ''}
                onClick={() => { setRightPanel('context'); setShowRightPanel(true) }}
                title="上下文管理"
              >
                📋
              </button>
              <button 
                className={showRightPanel && rightPanel === 'tools' ? 'active' : ''}
                onClick={() => { setRightPanel('tools'); setShowRightPanel(true) }}
                title="工具管理"
              >
                🔧
              </button>
              <button 
                className={showRightPanel && rightPanel === 'skills' ? 'active' : ''}
                onClick={() => { setRightPanel('skills'); setShowRightPanel(true) }}
                title="技能管理"
              >
                ⚡
              </button>
              <button 
                className={!showRightPanel ? 'active' : ''}
                onClick={() => setShowRightPanel(!showRightPanel)}
                title="隐藏面板"
              >
                ✕
              </button>
            </div>
          </div>
        </header>

        {/* 当前线程信息 */}
        <div className="thread-info-bar">
          <span className="thread-badge">
            当前对话: {currentThreadId}
          </span>
          <small>教程06：通过 thread_id 实现记忆隔离</small>
        </div>

        {/* 消息列表 */}
        <div className="messages">
          {messages.length === 0 && (
            <div className="empty-state">
              <p>👋 你好！我是 AI 助手</p>
              <p>试试问我：</p>
              <div className="suggestions">
                <button onClick={() => setInput('帮我算一下 123 乘以 456')}>计算 123 × 456</button>
                <button onClick={() => setInput('现在几点了？')}>查询时间</button>
                <button onClick={() => setInput('请记住：我喜欢简洁的回复')}>记住偏好</button>
              </div>
            </div>
          )}

          {messages.map((msg, i) => (
            <div key={i} className={`message ${msg.role}`}>
              <div className="message-avatar">
                {msg.role === 'user' ? '👤' : '🤖'}
              </div>
              <div className="message-body">
                {msg.toolCalls && msg.toolCalls.length > 0 && (
                  <div className="tool-calls">
                    {msg.toolCalls.map((tc, j) => (
                      <div key={j} className="tool-call">
                        <span className="tool-name">🔧 {tc.name}</span>
                        <span className="tool-args">({JSON.stringify(tc.args)})</span>
                        {tc.status === 'calling' && <span className="tool-status">执行中...</span>}
                        {tc.status === 'done' && <span className="tool-result">→ {tc.result}</span>}
                      </div>
                    ))}
                  </div>
                )}
                {msg.content && <div className="message-content">{msg.content}</div>}
                {msg.role === 'assistant' && loading && i === messages.length - 1 && !msg.content && (
                  <div className="typing">思考中...</div>
                )}
              </div>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>

        {/* 输入区域 */}
        <div className="input-area">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && sendMessage()}
            placeholder="输入消息..."
            disabled={loading}
          />
          <button onClick={sendMessage} disabled={loading || !input.trim()}>
            {loading ? '...' : '发送'}
          </button>
        </div>
      </div>

      {/* 右侧：管理面板 */}
      {showRightPanel && (
        <div className="right-panel">
          {renderRightPanel()}
        </div>
      )}
    </div>
  )
}

export default App
