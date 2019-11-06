#pragma once

#include <string>

// TODO custom assert macro

namespace ann
{
    // TODO impl class
    // ? err level
    class Status
    {
    public:
        enum error_codes
        {
            PLACEHOLDER = 0,
            INCOMPATIBLE_SIZES = 1
        };
    
        static Status OK();
        static Status ERROR(error_codes code, std::string message);

        bool ok() const;
        bool err() const;
        int code() const;
        const std::string& mesage() const;
        std::string to_string() const;

    private:
        int _code;
        std::string _message;

        // ? file, line & stacktrace
        // * boost might be useful for this
    };
}