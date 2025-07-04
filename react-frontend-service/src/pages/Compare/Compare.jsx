import React, {useContext, useEffect, useState} from "react";
import {fetchWithAuth} from "../../api/fetchWithAuth";
import {AuthContext} from "../AuthProvider";
import {useNavigate} from "react-router-dom";
import "./Compare.css"; // Создайте этот файл для стилей

export const Compare = () => {
    const [resumes, setResumes] = useState([]);
    const [vacancies, setVacancies] = useState([]);
    const [selectedResume, setSelectedResume] = useState(null);
    const [selectedVacancy, setSelectedVacancy] = useState(null);
    const [result, setResult] = useState(null);
    const [error, setError] = useState("");
    const {token} = useContext(AuthContext);
    const [isLoading, setIsLoading] = useState(false); // Для индикатора загрузки
    const [modalData, setModalData] = useState(null);   // Для хранения JSON ответа
    const [isModalOpen, setIsModalOpen] = useState(false); // Для управления видимостью модального окна

    const navigate = useNavigate();

    useEffect(() => {
        fetchResumes();
        fetchVacancies();
    }, []);

    const fetchResumes = async () => {
        try {
            const response = await fetchWithAuth("/resume/all", {method: "GET"}, token);
            const data = await response.json();
            setResumes(Array.isArray(data) ? data : []);
        } catch (err) {
            setError("Не удалось загрузить резюме");
            console.error(err);
        }
    };

    const fetchVacancies = async () => {
        try {
            const response = await fetchWithAuth("/vacancy/all", {method: "GET"}, token);
            const data = await response.json();
            setVacancies(Array.isArray(data) ? data : []);
        } catch (err) {
            setError("Не удалось загрузить вакансии");
            console.error(err);
        }
    };

    const handleCompare = async () => {
        // Убедитесь, что selectedResume, selectedVacancy и token доступны в этой области видимости
        // const { token } = useContext(AuthContext); // Если используете контекст для токена
        // const selectedResume = /* ваше значение */;
        // const selectedVacancy = /* ваше значение */;

        setIsLoading(true);
        setError(""); // Сбрасываем предыдущие ошибки
        setModalData(null); // Сбрасываем предыдущие данные модального окна

        try {
            const response = await fetchWithAuth(
                `/vacancy/compare?resume_id=${selectedResume}&vacancy_id=${selectedVacancy}`,
                {method: "GET"},
                token // Передаем токен, если он требуется функцией fetchWithAuth
            );

            if (!response.ok) {
                let errorMsg = "Ошибка сервера";
                try {
                    const errorData = await response.json(); // Попытка получить детали ошибки из JSON
                    errorMsg = errorData.message || errorData.error || `Ошибка ${response.status}`;
                } catch (e) {
                    // Оставить errorMsg как "Ошибка сервера" или использовать response.statusText
                    errorMsg = `Ошибка ${response.status}: ${response.statusText || "Не удалось получить детали ошибки"}`;
                }
                throw new Error(errorMsg);
            }

            const data = await response.json();
            setModalData(data);       // Сохраняем полученные данные
            setIsModalOpen(true);     // Открываем модальное окно
            // setResult(data);      // Если ваш старый setResult(data) делал что-то другое, адаптируйте
        } catch (err) {
            setError(err.message || "Произошла неизвестная ошибка.");
        } finally {
            setIsLoading(false);
        }
    };


    return (
        <div className="compare-page">
            <div className="compare-header">
                <h1>Сравнение резюме и вакансии</h1>
            </div>

            <div className="compare-content">
                <div className="selection-container">
                    {/* Секция с резюме */}
                    <div className="selection-section resumes-section">
                        <h2>Выберите резюме</h2>
                        <div className="items-list">
                            {resumes.length === 0 ? (
                                <p className="no-items">Нет доступных резюме</p>
                            ) : (
                                resumes.map(resume => (
                                    <div
                                        key={resume.id}
                                        className={`item-card ${selectedResume === resume.id ? 'selected' : ''}`}
                                        onClick={() => setSelectedResume(resume.id)}
                                    >
                                        <div className="radio-selector">
                                            <input
                                                type="radio"
                                                name="resume"
                                                checked={selectedResume === resume.id}
                                                onChange={() => {
                                                }}
                                            />
                                        </div>
                                        <div className="item-info">
                                            <h3>{resume.fileName}</h3>
                                            <p className="item-date">
                                                {new Date(resume.uploadedAt).toLocaleString()}
                                            </p>
                                        </div>
                                    </div>
                                ))
                            )}
                        </div>
                    </div>

                    {/* Секция с вакансиями */}
                    <div className="selection-section vacancies-section">
                        <h2>Выберите вакансию</h2>
                        <div className="items-list">
                            {vacancies.length === 0 ? (
                                <p className="no-items">Нет доступных вакансий</p>
                            ) : (
                                vacancies.map(vacancy => (
                                    <div
                                        key={vacancy.id}
                                        className={`item-card ${selectedVacancy === vacancy.id ? 'selected' : ''}`}
                                        onClick={() => setSelectedVacancy(vacancy.id)}
                                    >
                                        <div className="radio-selector">
                                            <input
                                                type="radio"
                                                name="vacancy"
                                                checked={selectedVacancy === vacancy.id}
                                                onChange={() => {
                                                }}
                                            />
                                        </div>
                                        <div className="item-info">
                                            <h3>{vacancy.title}</h3>
                                            <p className="item-date">
                                                {new Date(vacancy.createdAt).toLocaleString()}
                                            </p>
                                            <p className="item-url">
                                                <a
                                                    href={vacancy.url}
                                                    target="_blank"
                                                    rel="noopener noreferrer"
                                                    onClick={e => e.stopPropagation()}
                                                >
                                                    Ссылка на вакансию
                                                </a>
                                            </p>
                                        </div>
                                    </div>
                                ))
                            )}
                        </div>
                    </div>
                </div>

                {/* Кнопка сравнения */}
                <div className="compare-actions">
                    <button
                        className="compare-button"
                        onClick={handleCompare}
                        disabled={!selectedResume || !selectedVacancy}
                    >
                        {isLoading ? "Загрузка..." : "Сравнить и показать результат"}
                    </button>
                    {/* Отображение сообщения об ошибке, если есть */}
                    {error && <div style={{color: 'red', marginTop: '10px'}}>{error}</div>}

                    {/* Модальное окно для отображения JSON */}
                    {isModalOpen && modalData && (
                        <div className="modal-overlay"
                             onClick={() => setIsModalOpen(false) /* Закрытие по клику на оверлей */}>
                            <div className="modal-content"
                                 onClick={(e) => e.stopPropagation() /* Предотвращение закрытия при клике на контент */}>
                                <span className="modal-close" onClick={() => setIsModalOpen(false)}>&times;</span>
                                <h2>Результат запроса</h2>
                                <pre style={{
                                    maxHeight: '70vh', /* Ограничение высоты */
                                    overflowY: 'auto',  /* Прокрутка для длинного JSON */
                                    whiteSpace: 'pre-wrap', /* Перенос строк и сохранение форматирования */
                                    wordBreak: 'break-all', /* Перенос длинных строк без пробелов */
                                    background: '#f8f9fa',
                                    padding: '15px',
                                    border: '1px solid #dee2e6',
                                    borderRadius: '4px',
                                    textAlign: 'left'
                                }}>
                                    <div className="interview-result-detail">
                        <div className="result-summary">
                            <h2>Результат сравнения</h2>
                            <table>
                                <tbody>
                                <tr>
                                    <th>Процент соответствия</th>
                                    <td>{modalData.matchingPercent.toFixed(2)}%</td>
                                </tr>
                                <tr>
                                    <th>Недостающие навыки</th>
                                    <td>
                                        {modalData.neededSkills.slice(1, -1).replace(/'/g, "").replace(/, /g, "\n")}
                                    </td>
                                </tr>
                                    </tbody>
                            </table>
                        </div>
                    </div>
                    </pre>
                                <button
                                    onClick={() => setIsModalOpen(false)}
                                    style={{
                                        marginTop: '20px',
                                        padding: '10px 15px',
                                        backgroundColor: '#007bff',
                                        color: 'white',
                                        border: 'none',
                                        borderRadius: '4px',
                                        cursor: 'pointer'
                                    }}
                                >
                                    Закрыть
                                </button>
                            </div>
                        </div>
                    )}

                </div>

                {/* Результаты сравнения */}
                {result && (
                    <div className="comparison-results">
                        <h2>Результаты сравнения</h2>
                        <div className="result-card">
                            <div className="match-percentage">
                                <h3>Процент соответствия</h3>
                                <div className="percentage-display">
                                    {Math.round(result.matchPercentage)}%
                                </div>
                            </div>

                            <div className="skills-match">
                                <h3>Соответствие навыков</h3>
                                <ul>
                                    {Object.entries(result.skillsMatch).map(([skill, percentage]) => (
                                        <li key={skill}>
                                            {skill}: {percentage}%
                                        </li>
                                    ))}
                                </ul>
                            </div>

                            <div className="recommendations">
                                <h3>Рекомендации</h3>
                                {/* Проверяем, массив ли это или строка */}
                                {Array.isArray(result.recommendations) ? (
                                    <ul>
                                        {result.recommendations.map((item, index) => (
                                            <li key={index}>{item}</li>
                                        ))}
                                    </ul>
                                ) : (
                                    <p>{result.recommendations}</p>
                                )}
                            </div>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};