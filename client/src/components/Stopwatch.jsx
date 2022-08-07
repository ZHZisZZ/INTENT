import {useEffect, useState} from "react";

const Stopwatch = ({startTimestamp}) => {
    const [message, setMessage] = useState('')
    useEffect(() => {
        const update = setInterval(() => setMessage((Math.round((new Date() - startTimestamp) / 1000 * 10) / 10).toFixed(1) + 's'), 100)
        return () => clearTimeout(update)
    }, [])
    return message
}

export default Stopwatch
