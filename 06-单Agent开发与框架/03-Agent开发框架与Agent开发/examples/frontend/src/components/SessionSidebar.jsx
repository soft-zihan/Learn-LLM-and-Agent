import { useState, useEffect } from 'react'

/**
 * 会话侧边栏 - 对应教程06：记忆隔离
 * 
 * 功能：
 * - 显示所有对话线程列表
 * - 新建对话按钮
 * - 切换线程（thread_id）
 * - 删除线程
 * - 显示每个线程的消息数
 */

function SessionSidebar({ currentThreadId, onSelectThread, getApiUrl, backend }) {
  const [threads, setThreads] = useState([])
  const [loading, setLoading] = useState(false)

  // 加载线程列表
  useEffect(() => {
    loadThreads()
  }, [backend])

  const loadThreads = async () => {
    setLoading(true)
    try {
      const res = await fetch(`${getApiUrl(backend)}/api/threads`)
      const data = await res.json()
      setThreads(data.threads || [])
    } catch (err) {
      console.error('Failed to load threads:', err)
    } finally {
      setLoading(false)
    }
  }

  // 创建新线程
  const createThread = async () => {
    try {
      const res = await fetch(`${getApiUrl(backend)}/api/threads`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: `对话 ${threads.length + 1}` })
      })
      const data = await res.json()
      if (data.status === 'ok') {
        await loadThreads()
        onSelectThread(data.thread_id)
      }
    } catch (err) {
      console.error('Failed to create thread:', err)
    }
  }

  // 删除线程
  const deleteThread = async (threadId, e) => {
    e.stopPropagation()
    if (threadId === 'default') {
      alert('默认线程不能删除')
      return
    }
    if (!confirm('确定要删除这个对话吗？')) return
    
    try {
      const res = await fetch(`${getApiUrl(backend)}/api/threads/${threadId}`, {
        method: 'DELETE'
      })
      const data = await res.json()
      if (data.status === 'ok') {
        await loadThreads()
        if (currentThreadId === threadId) {
          onSelectThread('default')
        }
      }
    } catch (err) {
      console.error('Failed to delete thread:', err)
    }
  }

  return (
    <div className="session-sidebar">
      <div className="sidebar-header">
        <h3>对话列表</h3>
        <button className="new-chat-btn" onClick={createThread} title="新建对话">
          +
        </button>
      </div>
      
      <div className="thread-list">
        {loading ? (
          <div className="loading">加载中...</div>
        ) : threads.length === 0 ? (
          <div className="empty">暂无对话</div>
        ) : (
          threads.map(thread => (
            <div
              key={thread.thread_id}
              className={`thread-item ${thread.thread_id === currentThreadId ? 'active' : ''}`}
              onClick={() => onSelectThread(thread.thread_id)}
            >
              <div className="thread-info">
                <span className="thread-name">{thread.name}</span>
                <span className="thread-meta">
                  {thread.message_count > 0 && `${thread.message_count} 条消息`}
                </span>
              </div>
              {thread.thread_id !== 'default' && (
                <button 
                  className="delete-btn" 
                  onClick={(e) => deleteThread(thread.thread_id, e)}
                  title="删除"
                >
                  ×
                </button>
              )}
            </div>
          ))
        )}
      </div>
      
      <div className="sidebar-footer">
        <small>教程06：通过 thread_id 实现记忆隔离</small>
      </div>
    </div>
  )
}

export default SessionSidebar
