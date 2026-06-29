import { useState, useEffect } from 'react'

/**
 * 工具/MCP管理面板 - 对应教程03：MCP协议
 * 
 * 功能：
 * - 显示内置工具列表（带描述）
 * - 显示MCP工具列表
 * - MCP连接状态指示器（已连接/未连接）
 * 
 * 教程对应：
 * - 03-MCP协议：MCP客户端接入、工具发现
 */

function ToolsPanel({ getApiUrl, backend }) {
  const [tools, setTools] = useState([])
  const [mcpStatus, setMcpStatus] = useState({ connected: false, tools_count: 0 })
  const [mcpTools, setMcpTools] = useState([])
  const [loading, setLoading] = useState(false)

  // 加载工具列表
  useEffect(() => {
    loadTools()
  }, [backend])

  const loadTools = async () => {
    setLoading(true)
    try {
      // 加载普通工具
      const toolsRes = await fetch(`${getApiUrl(backend)}/api/tools`)
      const toolsData = await toolsRes.json()
      setTools(toolsData.tools || [])

      // 加载 MCP 状态
      const statusRes = await fetch(`${getApiUrl(backend)}/api/mcp/status`)
      const statusData = await statusRes.json()
      setMcpStatus(statusData)

      // 加载 MCP 工具详情
      if (statusData.connected) {
        const mcpRes = await fetch(`${getApiUrl(backend)}/api/mcp/tools`)
        const mcpData = await mcpRes.json()
        setMcpTools(mcpData.tools || [])
      }
    } catch (err) {
      console.error('Failed to load tools:', err)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="tools-panel">
      <div className="panel-header">
        <h3>工具管理</h3>
        <button className="refresh-btn" onClick={loadTools} title="刷新">
          🔄
        </button>
      </div>

      {loading ? (
        <div className="loading">加载中...</div>
      ) : (
        <>
          {/* MCP 状态 */}
          <div className="mcp-status-section">
            <div className="status-indicator">
              <span className={`status-dot ${mcpStatus.connected ? 'connected' : 'disconnected'}`} />
              <span className="status-text">
                MCP Server {mcpStatus.connected ? '已连接' : '未连接'}
              </span>
            </div>
            {mcpStatus.connected && (
              <span className="mcp-tools-count">
                {mcpStatus.tools_count} 个 MCP 工具
              </span>
            )}
          </div>

          {/* 内置工具 */}
          <div className="tools-section">
            <h4>内置工具</h4>
            <div className="tools-list">
              {tools.length === 0 ? (
                <div className="empty">暂无内置工具</div>
              ) : (
                tools.map((tool, i) => (
                  <div key={i} className="tool-item">
                    <div className="tool-name">{tool.name}</div>
                    <div className="tool-description">{tool.description}</div>
                  </div>
                ))
              )}
            </div>
          </div>

          {/* MCP 工具 */}
          {mcpStatus.connected && (
            <div className="tools-section">
              <h4>MCP 工具</h4>
              <div className="tools-list">
                {mcpTools.length === 0 ? (
                  <div className="empty">暂无 MCP 工具</div>
                ) : (
                  mcpTools.map((tool, i) => (
                    <div key={i} className="tool-item mcp-tool">
                      <div className="tool-name">
                        <span className="mcp-badge">MCP</span>
                        {tool.name}
                      </div>
                      <div className="tool-description">{tool.description}</div>
                    </div>
                  ))
                )}
              </div>
            </div>
          )}
        </>
      )}

      <div className="panel-footer">
        <small>教程03：MCP 协议实现工具标准化接入</small>
      </div>
    </div>
  )
}

export default ToolsPanel
