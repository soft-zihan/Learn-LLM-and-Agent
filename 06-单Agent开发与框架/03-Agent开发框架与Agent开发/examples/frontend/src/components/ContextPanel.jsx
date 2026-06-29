import { useState, useEffect } from 'react'

/**
 * 上下文管理面板 - 对应教程05：上下文工程
 * 
 * 功能：
 * - 显示当前上下文的消息列表
 * - 显示token使用统计（消息数、估算token数、使用率百分比）
 * - "压缩上下文"按钮（触发 summarize）
 * - 压缩前后对比
 * 
 * 教程对应：
 * - 05-提示词与上下文工程：2.3 上下文窗口管理策略
 */

function ContextPanel({ currentThreadId, getApiUrl, backend }) {
  const [stats, setStats] = useState(null)
  const [loading, setLoading] = useState(false)
  const [compressing, setCompressing] = useState(false)
  const [compressResult, setCompressResult] = useState(null)

  // 加载上下文统计
  useEffect(() => {
    if (currentThreadId) {
      loadContextStats()
    }
  }, [currentThreadId, backend])

  const loadContextStats = async () => {
    setLoading(true)
    try {
      const res = await fetch(`${getApiUrl(backend)}/api/context/${currentThreadId}`)
      const data = await res.json()
      setStats(data.stats)
    } catch (err) {
      console.error('Failed to load context stats:', err)
    } finally {
      setLoading(false)
    }
  }

  // 压缩上下文
  const compressContext = async () => {
    setCompressing(true)
    setCompressResult(null)
    try {
      const res = await fetch(`${getApiUrl(backend)}/api/context/${currentThreadId}/compress`, {
        method: 'POST'
      })
      const data = await res.json()
      setCompressResult(data)
      await loadContextStats()
    } catch (err) {
      console.error('Failed to compress context:', err)
      setCompressResult({ error: '压缩失败' })
    } finally {
      setCompressing(false)
    }
  }

  // 获取使用率颜色
  const getUsageColor = (percent) => {
    if (percent < 50) return '#4caf50'
    if (percent < 80) return '#ff9800'
    return '#f44336'
  }

  return (
    <div className="context-panel">
      <div className="panel-header">
        <h3>上下文管理</h3>
        <button className="refresh-btn" onClick={loadContextStats} title="刷新">
          🔄
        </button>
      </div>

      {loading ? (
        <div className="loading">加载中...</div>
      ) : stats ? (
        <>
          {/* 统计信息 */}
          <div className="stats-section">
            <div className="stat-item">
              <span className="stat-label">消息数</span>
              <span className="stat-value">{stats.message_count}</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">估算 Token</span>
              <span className="stat-value">{stats.estimated_tokens.toLocaleString()}</span>
            </div>
            <div className="stat-item">
              <span className="stat-label">使用率</span>
              <span className="stat-value" style={{ color: getUsageColor(stats.usage_percent) }}>
                {stats.usage_percent}%
              </span>
            </div>
          </div>

          {/* 使用率进度条 */}
          <div className="usage-bar-container">
            <div 
              className="usage-bar" 
              style={{ 
                width: `${Math.min(stats.usage_percent, 100)}%`,
                backgroundColor: getUsageColor(stats.usage_percent)
              }}
            />
          </div>

          {/* 角色分布 */}
          <div className="role-distribution">
            <h4>消息分布</h4>
            <div className="role-bars">
              {Object.entries(stats.role_counts).map(([role, count]) => (
                <div key={role} className="role-bar-item">
                  <span className="role-label">{role}</span>
                  <div className="role-bar-bg">
                    <div 
                      className="role-bar-fill"
                      style={{ width: `${stats.message_count > 0 ? (count / stats.message_count * 100) : 0}%` }}
                    />
                  </div>
                  <span className="role-count">{count}</span>
                </div>
              ))}
            </div>
          </div>

          {/* 警告提示 */}
          {stats.is_near_limit && (
            <div className="warning-box">
              ⚠️ 上下文接近限制，建议压缩
            </div>
          )}

          {/* 压缩按钮 */}
          <div className="compress-section">
            <button 
              className="compress-btn" 
              onClick={compressContext}
              disabled={compressing || stats.message_count < 5}
            >
              {compressing ? '压缩中...' : '压缩上下文'}
            </button>
            <small className="compress-hint">
              教程05：用 LLM 生成对话摘要，替代原始消息
            </small>
          </div>

          {/* 压缩结果 */}
          {compressResult && (
            <div className="compress-result">
              {compressResult.error ? (
                <div className="error">{compressResult.error}</div>
              ) : (
                <div className="success">
                  ✅ {compressResult.message}
                </div>
              )}
            </div>
          )}
        </>
      ) : (
        <div className="empty">选择对话以查看上下文</div>
      )}

      <div className="panel-footer">
        <small>教程05：上下文裁剪、压缩、优先级管理</small>
      </div>
    </div>
  )
}

export default ContextPanel
