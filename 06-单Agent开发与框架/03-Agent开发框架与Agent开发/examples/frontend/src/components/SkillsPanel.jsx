import { useState, useEffect } from 'react'

/**
 * 技能管理面板 - 对应教程04：Skills核心概念
 * 
 * 功能：
 * - 显示已加载的技能列表
 * - 技能详情（名称、描述）
 * 
 * 教程对应：
 * - 04-Skills核心概念：技能加载、注册、查询
 */

function SkillsPanel({ getApiUrl, backend }) {
  const [skills, setSkills] = useState([])
  const [loading, setLoading] = useState(false)

  // 加载技能列表
  useEffect(() => {
    loadSkills()
  }, [backend])

  const loadSkills = async () => {
    setLoading(true)
    try {
      const res = await fetch(`${getApiUrl(backend)}/api/skills`)
      const data = await res.json()
      setSkills(data.skills || [])
    } catch (err) {
      console.error('Failed to load skills:', err)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="skills-panel">
      <div className="panel-header">
        <h3>技能管理</h3>
        <button className="refresh-btn" onClick={loadSkills} title="刷新">
          🔄
        </button>
      </div>

      {loading ? (
        <div className="loading">加载中...</div>
      ) : (
        <div className="skills-list">
          {skills.length === 0 ? (
            <div className="empty">
              <p>暂无已加载的技能</p>
              <small>技能文件放在 skills/ 目录下</small>
            </div>
          ) : (
            skills.map((skill, i) => (
              <div key={i} className="skill-item">
                <div className="skill-header">
                  <span className="skill-name">{skill.name}</span>
                  <span className="skill-badge">Skill</span>
                </div>
                <div className="skill-description">{skill.description}</div>
                {skill.metadata && Object.keys(skill.metadata).length > 0 && (
                  <div className="skill-metadata">
                    {Object.entries(skill.metadata).map(([key, value]) => (
                      <span key={key} className="metadata-tag">
                        {key}: {value}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            ))
          )}
        </div>
      )}

      <div className="panel-footer">
        <small>教程04：Skills 是预定义的能力模块</small>
      </div>
    </div>
  )
}

export default SkillsPanel
